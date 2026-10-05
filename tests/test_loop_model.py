import copy
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from fastwam.loop.convert import build_experts, expert_configs, fold_blocks
from fastwam.loop.model import LoopWAM
from fastwam.loop.mot import LoopMoT
from fastwam.loop.schedule import PAIRS
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from test_loop_core import expert, inputs

TINY={'video':dict(hidden_dim=12,in_dim=2,out_dim=2,ffn_dim=24,text_dim=16,freq_dim=6,num_heads=2,attn_head_dim=6,patch_size=(1,1,1)),
      'action':dict(hidden_dim=8,action_dim=2,ffn_dim=16,text_dim=16,freq_dim=6,num_heads=2,attn_head_dim=6,cross_num_heads=1)}

class DummyVAE(nn.Module):
    temporal_downsample_factor=4

class CaptureTeacher(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(3,16)
        self.train_video_scheduler=WanContinuousFlowMatchScheduler(1000,5.)
        self.train_action_scheduler=WanContinuousFlowMatchScheduler(1000,5.)
    def _append_proprio_to_context(self,c,m,p):
        return torch.cat((c,self.proj(p).unsqueeze(1)),1),torch.cat((m,torch.ones(m.shape[0],1,dtype=torch.bool)),1)
    def _predict_joint_noise(self,v,a,tv,ta,c,m,f):
        self.seen=tuple(t.clone() for t in (v,a,tv,ta,c,m))
        return torch.zeros_like(v),torch.zeros_like(a)


def policy(recipe='L2',teacher=None):
    v,a=build_experts('loopwam',2,tiny_config=TINY)
    vc,ac=expert_configs('loopwam',TINY)
    meta=dict(arch='loopwam',lora_rank=2,video_config=vc,action_config=ac,proprio_dim=3)
    return LoopWAM(v,a,DummyVAE(),meta,training=True,loss_recipe=recipe,teacher=teacher)

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)

    def test_video_diagnostic_decodes_each_requested_exit(self):
        p=policy()
        x=torch.randn(1,2,3,2,2);a=torch.randn(1,4,2)
        tv=torch.tensor([200.]);ta=torch.tensor([300.]);c=torch.randn(1,4,16);mask=torch.ones(1,4,dtype=torch.bool)
        video,actions=p.denoise_configurations(x,a,tv,ta,c,mask,((4,4),(2,2),(1,1)),return_video_exits=True)
        self.assertEqual(set(video),{1,2,4})
        self.assertFalse(torch.equal(video[1],video[4]))
        self.assertFalse(torch.equal(video[2],video[4]))
        self.assertEqual(video[1].shape,x.shape)

    def test_prefix_and_first_frame_coda_at_all_exits(self):
        v,a=expert(),expert()
        v.blocks=fold_blocks(v.blocks,rank=3);a.blocks=fold_blocks(a.blocks,rank=3)
        m=LoopMoT(v,a)
        x,act,f,tv,ta,c,mask=inputs();cm=torch.ones(2,6,4,dtype=torch.bool)
        _,cache,exits=m.video_forward(x,f,tv,c,cm,mask,2,4)
        for kv in (1,2,3):
            _,short,shortex=m.video_forward(x,f,tv,c,cm,mask,2,kv)
            torch.testing.assert_close(exits[kv],shortex[kv],rtol=0,atol=0)
            m.add_exit_coda(cache,exits,kv,f,tv,c,cm,2)
            for j in range(3):
                for lhs,rhs in zip(cache['coda',kv,j],short['coda',kv,j]):
                    torch.testing.assert_close(lhs,rhs,rtol=1e-5,atol=1e-5)

    def test_end_to_end_velocity_full_rank_conversion_equality(self):
        vd,ad=build_experts('untied30',tiny_config=TINY)
        vc,ac=expert_configs('untied30',TINY)
        dense=LoopWAM(vd,ad,DummyVAE(),dict(arch='untied30',video_config=vc,action_config=ac,proprio_dim=3),training=True,loss_recipe='L2')
        v,a=copy.deepcopy(vd),copy.deepcopy(ad)
        v.blocks=fold_blocks(v.blocks,rank=32);a.blocks=fold_blocks(a.blocks,rank=32)
        loop=LoopWAM(v,a,DummyVAE(),dict(arch='loopwam',proprio_dim=3),training=True,loss_recipe='L2')
        x=torch.randn(2,2,3,2,2);noisy=torch.randn(2,4,2)
        tv=torch.tensor([200.,600.]);ta=torch.tensor([300.,800.]);c=torch.randn(2,4,16);mask=torch.ones(2,4,dtype=torch.bool)
        dv,da=dense.denoise_configurations(x,noisy,tv,ta,c,mask,((4,4),))
        lv,la=loop.denoise_configurations(x,noisy,tv,ta,c,mask,((4,4),))
        self.assertLess(float((dv-lv).abs().max()),1e-3)
        self.assertLess(float((da[4,4]-la[4,4]).abs().max()),1e-3)

    def test_teacher_receives_identical_noisy_inputs_and_own_context(self):
        teacher=CaptureTeacher();p=policy('L3',teacher)
        sample={'input_latents':torch.randn(2,2,3,2,2),'context':torch.randn(2,4,16),'context_mask':torch.ones(2,4,dtype=torch.bool),'proprio':torch.randn(2,4,3),'action':torch.randn(2,4,2)}
        original=p.denoise_configurations
        def capture(v,a,tv,ta,c,m,configs):
            self.student_seen=(v,a,tv,ta,c,m)
            return original(v,a,tv,ta,c,m,configs)
        p.denoise_configurations=capture
        loss,_=p(sample,global_step=0)
        for student,t in zip(self.student_seen[:4],teacher.seen[:4]):
            self.assertTrue(torch.equal(student,t))
        self.assertFalse(torch.equal(self.student_seen[4][:,-1],teacher.seen[4][:,-1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(param.grad is None for param in teacher.parameters()))
        missing=[n for n,param in p.named_parameters() if param.requires_grad and param.grad is None]
        self.assertEqual(missing,[])
        self.assertTrue(all(torch.isfinite(param.grad).all() for param in p.parameters() if param.grad is not None))

    def test_canonical_save_reload_every_budget(self):
        p=policy();p.mot.to(torch.bfloat16);p.proprio_encoder.to(torch.bfloat16)
        p.torch_dtype=torch.bfloat16;p.eval()
        q=policy();q.mot.to(torch.bfloat16);q.proprio_encoder.to(torch.bfloat16);q.torch_dtype=torch.bfloat16;q.eval()
        x=torch.randn(1,2,3,2,2,dtype=torch.bfloat16);a=torch.randn(1,4,2,dtype=torch.bfloat16)
        tv=torch.tensor([200.],dtype=torch.bfloat16);ta=torch.tensor([300.],dtype=torch.bfloat16);c=torch.randn(1,4,16,dtype=torch.bfloat16);mask=torch.ones(1,4,dtype=torch.bool)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.pt';p.export_checkpoint(path);q.load_checkpoint(path)
            self.assertFalse(any(n.startswith('teacher') or n.startswith('video_expert') for n in p.state_dict()))
            for pair in PAIRS:
                pv,pa=p.denoise_configurations(x,a,tv,ta,c,mask,(pair,))
                qv,qa=q.denoise_configurations(x,a,tv,ta,c,mask,(pair,))
                torch.testing.assert_close(pa[pair],qa[pair],rtol=0,atol=0)
            bad=torch.load(path,weights_only=False);del bad['action']['head.bias'];torch.save(bad,path)
            with self.assertRaises(RuntimeError):q.load_checkpoint(path)

if __name__=='__main__': unittest.main()
