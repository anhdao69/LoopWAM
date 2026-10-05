"""LoopWAM policy and exact-input teacher distillation.

Only the MoT and proprio encoder enter optimizer/checkpoints. Frozen VAE and
teacher are loaded from their original files. The teacher is deliberately not
registered as a student child module, so train()/DeepSpeed cannot mutate it.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch import nn
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .mot import LoopMoT
from .losses import action_loss, assert_shifts


class LoopWAM(FastWAM):
    def __init__(self,video,action,vae,meta,device='cpu',training=False,teacher=None,loss_recipe='L3',mode='fixed',seed=42,gradient_checkpointing=False):
        nn.Module.__init__(self)
        self.meta=dict(meta)
        self.device=torch.device(device)
        self.torch_dtype=torch.float32 if training else torch.bfloat16
        self.text_dim=int(video.text_embedding[0].in_features)
        self.proprio_dim=int(meta.get('proprio_dim',8))
        self.mot=LoopMoT(video,action,arch=meta['arch'],alignment=meta.get('alignment','late'),gradient_checkpointing=gradient_checkpointing)
        self.proprio_encoder=nn.Linear(self.proprio_dim,self.text_dim,device=device,dtype=self.torch_dtype)
        self.vae=vae
        self.text_encoder=self.tokenizer=None
        self.vae.requires_grad_(False).eval()
        for stream in ('video','action'):
            for phase in ('train','infer'):
                setattr(self,f'{phase}_{stream}_scheduler',WanContinuousFlowMatchScheduler(1000,5.))
        self.train_scheduler=self.train_video_scheduler
        self.infer_scheduler=self.infer_video_scheduler
        self.loss_lambda_video=self.loss_lambda_action=1.
        self.compile_training_denoise=False
        self.mode,self.seed,self.loss_recipe=mode,seed,loss_recipe
        if loss_recipe not in ('L2','L3'):
            raise ValueError('Initial campaign implements L2/L3 only')
        object.__setattr__(self,'teacher',teacher)
        if teacher is not None:
            teacher.requires_grad_(False).eval()
        self._prefill_graphs={}
        self._action_graphs={}
        self.train(training)
        self.check_shifts()

    @property
    def video_expert(self): return self.mot.mixtures['video']
    @property
    def action_expert(self): return self.mot.mixtures['action']
    @property
    def dit(self): return self.mot

    def train(self,mode=True):
        nn.Module.train(self,mode)
        self.vae.eval()
        return self

    def check_shifts(self):
        t=self.teacher
        assert_shifts(self.train_video_scheduler.shift,self.train_action_scheduler.shift,
                      t.train_video_scheduler.shift if t is not None else 5.,
                      t.train_action_scheduler.shift if t is not None else 5.)
        assert_shifts(self.infer_video_scheduler.shift,self.infer_action_scheduler.shift)

    @torch.no_grad()
    def _encode_video_latents(self,video_tensor,tiled=False,**kwargs):
        if tiled: raise ValueError('Training VAE tiling is not enabled')
        # A frozen deterministic encoder; an on-disk latent cache may supply
        # input_latents directly. Avoid cudagraph private buffers in training.
        return self.vae.model.encode(video_tensor.to(self.device),self.vae.scale)

    def set_budget(self,kv,ka):
        old=(self.mot.kv,self.mot.ka)
        if hasattr(self,'_prefill_video_cache_compiled'):
            self._prefill_graphs[old[0]]=self._prefill_video_cache_compiled
            del self._prefill_video_cache_compiled
        if hasattr(self,'_denoise_action_with_video_cache_compiled'):
            self._action_graphs[old]=self._denoise_action_with_video_cache_compiled
            del self._denoise_action_with_video_cache_compiled
        self.mot.set_budget(kv,ka)
        if kv in self._prefill_graphs:
            self._prefill_video_cache_compiled=self._prefill_graphs[kv]
        if (kv,ka) in self._action_graphs:
            self._denoise_action_with_video_cache_compiled=self._action_graphs[kv,ka]

    @torch.no_grad()
    def infer_action(self,prompt,input_image,action_horizon,proprio=None,context=None,context_mask=None,negative_prompt=None,text_cfg_scale=1.,num_inference_steps=10,sigma_shift=None,seed=None,rand_device='cpu',tiled=False,compile_action_infer=False,Kv=None,Ka=None):
        if sigma_shift is not None and float(sigma_shift)!=5.:
            raise ValueError('LoopWAM inference shift must be 5.0')
        if Kv is not None or Ka is not None:
            self.set_budget(self.mot.kv if Kv is None else Kv,self.mot.ka if Ka is None else Ka)
        return super().infer_action(prompt,input_image,action_horizon,proprio,context,context_mask,negative_prompt,text_cfg_scale,num_inference_steps,5.,seed,rand_device,tiled,compile_action_infer)

    def _inputs(self,sample):
        move=lambda x,dtype=None:x.to(device=self.device,dtype=dtype,non_blocking=True)
        if 'input_latents' in sample:
            latents=move(sample['input_latents'],self.torch_dtype)
        else:
            latents=self._encode_video_latents(move(sample['video'],self.torch_dtype))
        context=move(sample['context'],self.torch_dtype)
        mask=move(sample['context_mask'],torch.bool)
        proprio=move(sample['proprio'],self.torch_dtype)
        if proprio.ndim==3: proprio=proprio[:,0]
        action=move(sample['action'],self.torch_dtype)
        return latents,context,mask,proprio,action

    def denoise_configurations(self,latents,noisy_action,tv,ta,context,context_mask,configurations):
        vp=self.video_expert.prepare(x=latents,timestep=tv,context=context,context_mask=context_mask,action=None,fuse_vae_embedding_in_latents=True)
        vx,t,tm,vc,vcm,vf,frames,height,width,first=vp
        vmask=self.video_expert.build_video_to_video_mask(vx.shape[1],first,vx.device)
        vx,cache,exits=self.mot.video_forward(vx,vf,tm,vc,vcm,vmask,first,kv=4)
        video=self.video_expert.post(vx,t,frames,height,width)
        ax,_,atm,ac,acm,af=self.action_expert.prepare(noisy_action,ta,context,context_mask)
        actions={}
        for kv,ka in configurations:
            self.mot.add_exit_coda(cache,exits,kv,vf,tm,vc,vcm,first)
            actions[kv,ka]=self.action_expert.post(self.mot.action_forward(ax,af,atm,ac,acm,cache,kv,ka))
        return video,actions

    def forward(self,sample,global_step=0):
        from .sampler import configurations_for_step
        self.check_shifts()
        clean,base_ctx,base_mask,proprio,action=self._inputs(sample)
        context,context_mask=self._append_proprio_to_context(base_ctx,base_mask,proprio)
        b=action.shape[0]
        noisev=torch.randn_like(clean); noisea=torch.randn_like(action)
        tv=self.train_video_scheduler.sample_training_t(b,self.device,clean.dtype)
        ta=self.train_action_scheduler.sample_training_t(b,self.device,action.dtype)
        latents=self.train_video_scheduler.add_noise(clean,noisev,tv)
        latents[:,:,0:1]=clean[:,:,0:1]
        noisy_action=self.train_action_scheduler.add_noise(action,noisea,ta)
        av=noisea-action; vv=noisev-clean
        teacher_action=None
        if self.loss_recipe=='L3':
            if self.teacher is None: raise ValueError('L3 requires a frozen teacher')
            with torch.no_grad():
                tc,tm=self.teacher._append_proprio_to_context(base_ctx,base_mask,proprio)
                _,teacher_action=self.teacher._predict_joint_noise(latents,noisy_action,tv,ta,tc,tm,True)
        configs=configurations_for_step(global_step,self.mode,self.seed) if self.meta['arch']=='loopwam' else ((4,4),)
        video,actions=self.denoise_configurations(latents,noisy_action,tv,ta,context,context_mask,configs)
        vpad=sample.get('image_is_pad')
        if vpad is not None: vpad=vpad.to(self.device,dtype=torch.bool)
        lv=(self._compute_video_loss_per_sample(video[:,:,1:],vv[:,:,1:],vpad,False)*self.train_video_scheduler.training_weight(tv)).mean()
        total=lv
        metrics={'video_fm':lv.detach()}
        apad=sample.get('action_is_pad')
        if apad is not None: apad=apad.to(self.device,dtype=torch.bool)
        augmented=sample.get('is_augmented')
        aw=self.train_action_scheduler.training_weight(ta)
        for pair,pred in actions.items():
            key=f'{pair[0]}_{pair[1]}'
            fm=action_loss(pred,av,aw,apad)
            total=total+fm
            metrics[f'action_fm/{key}']=fm.detach()
            if teacher_action is not None:
                kd=action_loss(pred,teacher_action,aw,apad,augmented)
                total=total+kd
                metrics[f'action_kd/{key}']=kd.detach()
        metrics['loss']=total.detach()
        return total,metrics

    def training_loss(self,sample,tiled=False):
        return self.forward(sample)

    def export_checkpoint(self,path,use_state=None,step=None):
        state=self.state_dict()
        if use_state is not None:
            expected={n for n,p in self.named_parameters() if p.requires_grad}
            if set(use_state)!=expected:
                raise ValueError(f'EMA parameter mismatch: missing={expected-set(use_state)}, extra={set(use_state)-expected}')
            state.update(use_state)
        def extract(prefix):
            return {k[len(prefix):]:v.detach().to(device='cpu',dtype=torch.bfloat16).clone() for k,v in state.items() if k.startswith(prefix)}
        payload={'format':'loopwam_v1','meta':dict(self.meta),'step':step,
                 'video':extract('mot.mixtures.video.'),'action':extract('mot.mixtures.action.'),'proprio':extract('proprio_encoder.')}
        payload['meta'].update(loss_recipe=self.loss_recipe,sigma_shift=5.,mode=self.mode,seed=self.seed)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,temporary);os.replace(temporary,path)

    def load_checkpoint(self,path,optimizer=None):
        if optimizer is not None: raise ValueError('Restore optimizer from trainer state, not policy export')
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if payload.get('format')!='loopwam_v1': raise ValueError('Not a canonical LoopWAM checkpoint')
        if payload['meta']['arch']!=self.meta['arch']: raise ValueError('Architecture mismatch')
        self.video_expert.load_state_dict(payload['video'],strict=True)
        self.action_expert.load_state_dict(payload['action'],strict=True)
        self.proprio_encoder.load_state_dict(payload['proprio'],strict=True)
        return payload


def _load_vae(device):
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    root=Path(os.environ.get('DIFFSYNTH_MODEL_BASE_PATH','checkpoints'))
    return _load_registered_model(str(root/'Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'),'wan_video_vae',torch.bfloat16,str(device)).requires_grad_(False).eval()


def load_teacher(checkpoint,device='cuda',vae=None):
    from omegaconf import OmegaConf
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    cfg=OmegaConf.to_container(OmegaConf.load(Path(__file__).resolve().parents[3]/'configs/model/fastwam.yaml'),resolve=False)
    vc=dict(cfg['video_dit_config']);ac=dict(cfg['action_dit_config'])
    vc.update(action_dim=7,use_gradient_checkpointing=False)
    ac.update(action_dim=7,use_gradient_checkpointing=False)
    # Instantiate CPU directly in bf16 to avoid a temporary 24GB fp32 teacher.
    with torch.device('meta'):
        video=WanVideoDiT(**vc);action=ActionDiT(**ac)
    # Frequency tables are ordinary tensors and must be reconstructed from CPU.
    from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis,precompute_freqs_cis_3d
    video.freqs=precompute_freqs_cis_3d(video.attn_head_dim)
    action.freqs=precompute_freqs_cis(action.attn_head_dim,1024)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    for name,expert in (('video',video),('action',action)):
        prefix=f'mixtures.{name}.'
        state={k[len(prefix):]:v for k,v in payload['mot'].items() if k.startswith(prefix)}
        expert.load_state_dict(state,strict=True,assign=True)
        expert.to(device=device,dtype=torch.bfloat16)
    mot=MoT({'video':video,'action':action})
    model=FastWAM(video,action,mot,vae if vae is not None else _load_vae(device),text_dim=4096,proprio_dim=8,device=str(device),torch_dtype=torch.bfloat16)
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
    model.requires_grad_(False).eval()
    return model


def load_model(checkpoint,device='cuda',training=False,teacher_checkpoint=None,loss_recipe='L3',mode='fixed',seed=42,gradient_checkpointing=False):
    from .convert import build_experts
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('format')!='loopwam_v1':
        if training: raise ValueError('Training init must be a converted LoopWAM checkpoint')
        return load_teacher(checkpoint,device)
    meta=payload['meta']
    if float(meta.get('sigma_shift',5.))!=5.: raise ValueError('Checkpoint shift must be 5')
    dtype=torch.float32 if training else torch.bfloat16
    video,action=build_experts(arch=meta['arch'],lora_rank=meta.get('lora_rank',32),device='cpu',dtype=dtype,tiny_config={'video':meta['video_config'],'action':meta['action_config']})
    video.load_state_dict(payload['video'],strict=True);action.load_state_dict(payload['action'],strict=True)
    video.to(device=device,dtype=dtype);action.to(device=device,dtype=dtype)
    vae=_load_vae(device)
    teacher=load_teacher(teacher_checkpoint,device,vae) if training and loss_recipe=='L3' and teacher_checkpoint else None
    model=LoopWAM(video,action,vae,meta,device,training,teacher,loss_recipe,mode,seed,gradient_checkpointing)
    model.proprio_encoder.load_state_dict(payload['proprio'],strict=True)
    return model
