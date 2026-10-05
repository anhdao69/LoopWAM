"""Fixed-noise open-loop validation on an immutable, twenty-demonstration panel."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from torch.utils.data import default_collate
from .model import load_teacher
from .losses import action_loss


def panel_indices(dataset):
    """One midpoint unpadded clip per held-out demo, stable across all runs.

    This is a diagnostic panel, not a claim to cover all 5,438 validation
    windows. Saving original IDs makes every measured clip reproducible.
    """
    positions={window:index for index,window in enumerate(dataset.indices)}
    selected=[]
    for episode in dataset.manifest['episodes']:
        if episode['split']=='validation':
            if episode['unpadded_windows']<1:
                raise ValueError('Validation demonstrations must contain an unpadded clip')
            window=episode['window_start']+(episode['unpadded_windows']-1)//2
            selected.append(positions[window])
    if len(selected)!=20:
        raise ValueError('Expected exactly twenty held-out demonstrations')
    return selected


def diagnostic_pairs(model,step):
    if model.meta['arch']!='loopwam' or model.mode=='fixed': return ((4,4),)
    if model.mode=='konly': return ((4,4),(4,2),(4,1))
    if model.mode=='coupled' or (model.mode=='three_stage' and step<14000): return ((4,4),(2,2),(1,1))
    return ((4,4),(4,2),(4,1),(2,2),(1,1))


@torch.no_grad()
def _sample_actions(model,first_frame,context,mask,pair,seed=1234):
    kv,ka=pair
    vp=model.video_expert.prepare(first_frame,torch.zeros(1,device=first_frame.device,dtype=first_frame.dtype),context,mask,action=None,fuse_vae_embedding_in_latents=True)
    x,_,tm,c,cm,f,_,_,_,first=vp
    vmask=torch.ones((first,first),device=x.device,dtype=torch.bool)
    _,cache,_=model.mot.video_forward(x,f,tm,c,cm,vmask,first,kv)
    generator=torch.Generator(device=first_frame.device).manual_seed(seed)
    a=torch.randn((1,32,model.action_expert.action_dim),generator=generator,device=first_frame.device,dtype=torch.float32)
    timesteps,deltas=model.infer_action_scheduler.build_inference_schedule(10,first_frame.device,a.dtype,5.)
    for t,delta in zip(timesteps,deltas):
        ax,_,am,ac,acm,af=model.action_expert.prepare(a,t.reshape(1),context,mask)
        velocity=model.action_expert.post(model.mot.action_forward(ax,af,am,ac,acm,cache,kv,ka))
        a=model.infer_action_scheduler.step(velocity,delta,a)
    return a


@torch.no_grad()
def run_open_loop(model,validation_dataset,output_dir,global_step,teacher_checkpoint,seed=1234):
    """OL1 velocity MSE, OL2 first-ten L1, OL3 future-video velocity MSE.

    Called on rank zero between optimizer steps under the caller's selected raw
    or EMA weights. This routine itself does not swap weights or step anything.
    """
    start=time.monotonic()
    training=model.training
    model.eval()
    teacher=model.teacher
    temporary=teacher is None
    if temporary: teacher=load_teacher(teacher_checkpoint,model.device,model.vae)
    pairs=diagnostic_pairs(model,global_step)
    totals={f'{v}_{a}':{'ol1':0.,'ol2':0.,'ol3':0.} for v,a in pairs}
    windows=[]
    try:
        with torch.random.fork_rng(devices=[model.device.index or 0]),torch.autocast('cuda',dtype=torch.bfloat16):
            for panel_id,index in enumerate(panel_indices(validation_dataset)):
                sample=default_collate([validation_dataset[index]])
                windows.append(int(sample['window_id'][0]))
                clean,base,mask,proprio,action=model._inputs(sample)
                context,cm=model._append_proprio_to_context(base,mask,proprio)
                tc,tcm=teacher._append_proprio_to_context(base,mask,proprio)
                generator=torch.Generator(device=model.device).manual_seed(seed+panel_id)
                nv=torch.randn(clean.shape,generator=generator,device=model.device,dtype=clean.dtype)
                na=torch.randn(action.shape,generator=generator,device=model.device,dtype=action.dtype)
                apad=sample['action_is_pad'].to(model.device)
                vpad=sample['image_is_pad'].to(model.device)
                for tau in (.1,.3,.5,.7,.9):
                    t=torch.tensor([tau*1000],device=model.device,dtype=torch.float32)
                    x=model.train_video_scheduler.add_noise(clean,nv,t);x[:,:,0:1]=clean[:,:,0:1]
                    a=model.train_action_scheduler.add_noise(action,na,t)
                    tv,ta=teacher._predict_joint_noise(x,a,t,t,tc,tcm,True)
                    sv,sa=model.denoise_configurations(x,a,t,t,context,cm,pairs,return_video_exits=True)
                    for pair in pairs:
                        row=totals[f'{pair[0]}_{pair[1]}']
                        row['ol1']+=float(action_loss(sa[pair],ta,torch.ones(1,device=model.device),apad))/5
                        row['ol3']+=float(model._compute_video_loss_per_sample(sv[pair[0]][:,:,1:],tv[:,:,1:],vpad,False).mean())/5
                for pair in pairs:
                    pred=_sample_actions(model,clean[:,:,:1],context,cm,pair,seed+panel_id)
                    error=(pred[:,:10].float()-action[:,:10].float()).abs().mean(-1)
                    valid=(~apad[:,:10]).float()
                    totals[f'{pair[0]}_{pair[1]}']['ol2']+=float((error*valid).sum()/valid.sum().clamp_min(1.))
        for row in totals.values():
            for key in row: row[key]/=len(windows)
        record=dict(global_step=global_step,panel='one fixed unpadded midpoint clip per held-out demonstration',window_ids=windows,
                    tau_grid=[.1,.3,.5,.7,.9],seed=seed,metrics=totals,seconds=time.monotonic()-start)
        path=Path(output_dir)/'open_loop';path.mkdir(parents=True,exist_ok=True)
        target=path/f'step_{global_step:08d}.json'
        temporary_path=target.with_suffix('.json.tmp');temporary_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary_path.replace(target)
        return record
    finally:
        model.train(training)
        if temporary:
            del teacher
            torch.cuda.empty_cache()
