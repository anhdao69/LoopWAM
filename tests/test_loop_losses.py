import unittest
import torch
from fastwam.loop.losses import action_loss, assert_shifts

class LossTests(unittest.TestCase):
    def test_kd_masks_padding_and_augmented_samples(self):
        p=torch.tensor([[[2.],[99.]],[[3.],[3.]]],requires_grad=True)
        target=torch.zeros_like(p)
        pad=torch.tensor([[False,True],[False,False]])
        loss=action_loss(p,target,torch.tensor([2.,1.]),pad,torch.tensor([False,True]))
        self.assertEqual(float(loss),4.)
        loss.backward()
        self.assertEqual(float(p.grad[0,1]),0.)
        self.assertTrue(torch.equal(p.grad[1],torch.zeros_like(p.grad[1])))
    def test_all_padding_finite_zero(self):
        p=torch.ones(2,3,7,requires_grad=True)
        loss=action_loss(p,torch.zeros_like(p),torch.ones(2),torch.ones(2,3,dtype=torch.bool))
        self.assertEqual(float(loss),0.)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
    def test_shift_guard(self):
        assert_shifts(5.,5.,5.,5.)
        for shifts in ((1.,5.,5.,5.),(5.,5.,1.,5.),(1.,1.,1.,1.)):
            with self.assertRaises(ValueError): assert_shifts(*shifts)

if __name__=='__main__': unittest.main()
