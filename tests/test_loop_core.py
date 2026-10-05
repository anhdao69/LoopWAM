"""Numerical invariants for the actual Wan attention/residual kernels."""
import copy
import unittest
import torch
from torch import nn
from fastwam.models.wan22.wan_video_dit import DiTBlock, precompute_freqs_cis
from fastwam.models.wan22.mot import MoT
from fastwam.loop.mot import LoopMoT
from fastwam.loop.schedule import action_schedule, video_schedule, PAIRS


def expert(depth=30):
    m = nn.Module()
    m.num_heads, m.attn_head_dim = 2, 6
    m.use_gradient_checkpointing = False
    m.blocks = nn.ModuleList([DiTBlock(12, 6, 2, 24) for _ in range(depth)])
    return m


def inputs():
    torch.manual_seed(42)
    v, a = torch.randn(2, 6, 12), torch.randn(2, 3, 12)
    freq = precompute_freqs_cis(6, 6).view(6, 1, 3)
    tv, ta = torch.randn(2, 6, 6, 12), torch.randn(2, 6, 12)
    # The first frame always has clean time modulation.
    tv[:, :2] = 0
    ctx = torch.randn(2, 4, 12)
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[:2, 2:] = False
    return v, a, freq, tv, ta, ctx, mask


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_schedule_all_pairs_and_suffix_slots(self):
        self.assertEqual(len(PAIRS), 10)
        for kv, ka in PAIRS:
            video = video_schedule(kv)
            action = action_schedule(kv, ka)
            self.assertEqual(len(video), 6 + 6 * kv)
            self.assertEqual(len(action), 6 + 6 * ka)
            keys = {entry[2] for entry in video}
            self.assertTrue(all(entry[2] in keys for entry in action))
            core = action[3:-3]
            self.assertEqual([entry[1] for entry in core], [r for r in range(kv-ka,kv) for _ in range(6)])
        with self.assertRaises(ValueError):
            action_schedule(2, 3)

    def test_dense_separate_stream_equals_joint_attention(self):
        ve, ae = expert(), expert()
        joint = MoT({'video':copy.deepcopy(ve), 'action':copy.deepcopy(ae)})
        loop = LoopMoT(ve, ae, arch='untied30')
        v,a,f,tv,ta,c,m = inputs()
        jointmask = torch.zeros(9,9,dtype=torch.bool)
        jointmask[:6,:6]=m; jointmask[6:,:2]=True; jointmask[6:,6:]=True
        args = dict(video_tokens=v,action_tokens=a,video_freqs=f,action_freqs=f[:3],video_t_mod=tv,action_t_mod=ta,video_context=c,video_context_mask=torch.ones(2,6,4,dtype=torch.bool),action_context=c,action_context_mask=torch.ones(2,3,4,dtype=torch.bool),attention_mask=jointmask)
        jv,ja=joint.forward_joint_core(**args)
        lv,la=loop.forward_joint_core(**args)
        torch.testing.assert_close(lv,jv,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(la,ja,rtol=1e-5,atol=1e-5)

    def test_cache_causality_and_future_isolation(self):
        loop=LoopMoT(expert(),expert(),arch='untied30')
        v,a,f,tv,ta,c,m=inputs()
        cm=torch.ones(2,6,4,dtype=torch.bool)
        full,cache,_=loop.video_forward(v,f,tv,c,cm,m,tokens_per_frame=2,kv=4)
        _,short,_=loop.video_forward(v[:,:2],f[:2],tv[:,:2],c,cm[:,:2],m[:2,:2],tokens_per_frame=2,kv=4)
        for key in cache:
            for x,y in zip(cache[key],short[key]):
                torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
        changed=v.clone();changed[:,2:]=torch.randn_like(changed[:,2:])*20
        _,other,_=loop.video_forward(changed,f,tv,c,cm,m,tokens_per_frame=2,kv=4)
        for key in cache:
            torch.testing.assert_close(cache[key][0],other[key][0],rtol=1e-5,atol=1e-5)
        action_mask=torch.ones(2,3,4,dtype=torch.bool)
        expected=loop.action_forward(a,f[:3],ta,c,action_mask,cache,4,4)
        actual=loop.action_forward(a,f[:3],ta,c,action_mask,other,4,4)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        # A preceding training action branch must not leave label-dependent state
        # in the first-frame cache used by fixed-noise inference.
        loop.action_forward(a*100,f[:3],ta,c,action_mask,cache,4,4)
        after=loop.action_forward(a,f[:3],ta,c,action_mask,cache,4,4)
        torch.testing.assert_close(after,expected,rtol=0,atol=0)

    def test_checkpointed_gradient_equals_uncheckpointed(self):
        first=LoopMoT(expert(12),expert(12),arch='untied12')
        second=copy.deepcopy(first);second.gradient_checkpointing=True
        v,a,f,tv,ta,c,m=inputs()
        jm=torch.zeros(9,9,dtype=torch.bool);jm[:6,:6]=m;jm[6:,:2]=True;jm[6:,6:]=True
        args=dict(video_tokens=v,action_tokens=a,video_freqs=f,action_freqs=f[:3],video_t_mod=tv,action_t_mod=ta,video_context=c,video_context_mask=torch.ones(2,6,4,dtype=torch.bool),action_context=c,action_context_mask=torch.ones(2,3,4,dtype=torch.bool),attention_mask=jm)
        for obj in (first,second):
            vo,ao=obj.forward_joint_core(**args);(vo.square().mean()+ao.square().mean()).backward()
        for (n,p),(n2,q) in zip(first.named_parameters(),second.named_parameters()):
            self.assertEqual(n,n2);self.assertIsNotNone(p.grad,n);self.assertIsNotNone(q.grad,n)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-5)

if __name__=='__main__': unittest.main()
