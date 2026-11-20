import unittest

import torch

from src.assigner import assign_image, pairwise_iou
from src.metrics import class_aware_nms, nms
from src.loss import raw_box_to_xyxy


class CoreTests(unittest.TestCase):
    def test_iou_identity(self):
        b = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
        self.assertAlmostEqual(float(pairwise_iou(b, b)[0, 0]), 1.0, places=5)

    def test_dual_assignment_topk(self):
        grid = 4
        boxes = torch.tensor([[0.15, 0.15, 0.35, 0.35]]).repeat(grid * grid, 1)
        gts, labels = torch.tensor([[0.25, 0.25, 0.2, 0.2]]), torch.tensor([1])
        obj = torch.zeros(1, grid * grid)
        cls = torch.zeros(1, grid * grid, 3)
        # Make the top-left region attractive but leave many candidates.
        obj[0, :] = -2.0
        obj[0, 5] = 2.0
        cls[0, :, 1] = 1.0
        a_many = assign_image(boxes, obj[0], cls[0], gts, labels, grid=grid, topk=3)
        a_one = assign_image(boxes, obj[0], cls[0], gts, labels, grid=grid, topk=1)
        self.assertEqual(len(a_many), 3)
        self.assertEqual(len(a_one), 1)
        self.assertEqual(len({a.cell_idx for a in a_many}), 3)

    def test_nms_suppresses_duplicate(self):
        boxes = torch.tensor(
            [
                [0.10, 0.10, 0.40, 0.40],
                [0.11, 0.11, 0.41, 0.41],
                [0.70, 0.70, 0.95, 0.95],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [0, 2])

    def test_class_aware_nms(self):
        pred = {
            "boxes": torch.tensor(
                [[0.1, 0.1, 0.4, 0.4], [0.11, 0.11, 0.41, 0.41], [0.1, 0.1, 0.4, 0.4]]
            ),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([0, 0, 1]),
        }
        out = class_aware_nms(pred, 0.5)
        self.assertEqual(len(out["scores"]), 2)

    def test_box_decode_range(self):
        raw = torch.zeros(2, 16, 4)
        boxes = raw_box_to_xyxy(raw, 4)
        self.assertTrue(bool((boxes >= 0).all() and (boxes <= 1).all()))


if __name__ == "__main__":
    unittest.main()
