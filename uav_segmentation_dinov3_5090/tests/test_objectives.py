import unittest
from unittest.mock import patch
import torch
from aic.objectives import edge_targets,boundary_loss,iou_surrogate
from aic.model import Decoder


class ObjectiveTests(unittest.TestCase):
    def test_straight_boundary(self):
        target=torch.zeros(1,6,8,dtype=torch.long); target[:,:,4:]=1
        edges,safe=edge_targets(target)
        expected=torch.zeros_like(edges); expected[:,:,3:5]=1
        self.assertTrue(torch.equal(edges,expected)); self.assertTrue(safe.all())

    def test_ignore_hole_not_an_edge(self):
        target=torch.zeros(1,7,7,dtype=torch.long); target[:,3,3]=-1
        edge,safe=edge_targets(target)
        self.assertEqual(edge.sum().item(),0)
        self.assertFalse(safe[:,2:5,2:5].any())
        logits=torch.zeros(1,1,7,7,requires_grad=True)
        boundary_loss(logits,target).backward()
        self.assertEqual(logits.grad[:,:,2:5,2:5].abs().sum().item(),0)

    def test_boundary_degenerate_crops(self):
        for value in [-1,255,0]:
            target=torch.full((1,5,5),value,dtype=torch.long)
            scores=torch.randn(1,1,5,5,requires_grad=True)
            loss=boundary_loss(scores,target); loss.backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            if value in [-1,255]: self.assertEqual(loss.item(),0)
            else: self.assertGreater(loss.item(),0)

    def test_lovasz_perfect_wrong_ignore(self):
        target=torch.zeros(1,4,6,dtype=torch.long); target[:,:,3:]=1
        good=torch.nn.functional.one_hot(target,8).permute(0,3,1,2).float()*40-20
        self.assertLess(iou_surrogate(good,target).item(),1e-6)
        self.assertGreater(iou_surrogate(good.roll(1,1),target).item(),.9)
        scores=torch.randn(1,8,4,6,requires_grad=True)
        target.fill_(255); loss=iou_surrogate(scores,target); loss.backward()
        self.assertEqual(loss.item(),0); self.assertEqual(scores.grad.abs().sum().item(),0)

    def test_auxiliary_gradient_and_inference_skip(self):
        decoder=Decoder(32,32,boundary=True)
        features=[torch.randn(1,32,4,4) for _ in range(4)]
        image=torch.randn(1,3,56,56)
        scores,edges=decoder(features,image,return_boundary=True)
        target=torch.randint(0,8,(1,56,56))
        loss=boundary_loss(edges,target)+iou_surrogate(scores,target)
        loss.backward()
        self.assertGreater(decoder.boundary_head.weight.grad.abs().sum().item(),0)
        self.assertGreater(decoder.refine[0].weight.grad.abs().sum().item(),0)
        with patch.object(decoder.boundary_head,'forward',side_effect=AssertionError('must not run at inference')):
            self.assertEqual(decoder(features,image).shape,(1,8,56,56))


if __name__=='__main__': unittest.main()
