import itertools
import torch
from pytorch_lightning.metrics import ConfusionMatrix, Metric
from torchvision.ops import box_iou


class MyConfusionMatrix(ConfusionMatrix):
    @staticmethod
    def _get_highest_score_label(predicted_boxes, iou_scores):
        iou_filter = (iou_scores > 0.5).squeeze()
        predictions_per_ground_truth = predicted_boxes[iou_filter]
        # TODO: nondeterministic output of argmax for equal scores (rare?)
        if not predictions_per_ground_truth.numel(): return 0  # if predictions are empty return background class
        prediction_scores = predictions_per_ground_truth.narrow(-1, start=0, length=1)
        prediction_labels = predictions_per_ground_truth.narrow(-1, start=1, length=1).int()
        best_score_indice = prediction_scores.argmax()
        return prediction_labels[best_score_indice].item()

    def update(self, records, preds):
        batch_target_labels = []
        batch_labels = []
        # matching detections with targets
        for image_targets, image_preds in zip(records, preds):
            target_boxes = torch.Tensor([bbox.xyxy for bbox in image_targets['bboxes']])
            predicted_bboxes = torch.stack([
                torch.Tensor([score, label, *bbox.xyxy])  # bind scores, labels and boxes
                for score, label, bbox in zip(image_preds['scores'], image_preds['labels'], image_preds['bboxes'])
            ])
            iou_scores = box_iou(predicted_bboxes.narrow(-1, 2, 4), target_boxes)
            ious_per_ground_truth = iou_scores.split(split_size=1, dim=1)
            labels = [self._get_highest_score_label(predicted_bboxes, iou) for iou in ious_per_ground_truth]

            batch_labels.append(labels)
            batch_target_labels.append(image_targets['labels'])

        assert(len(batch_labels) == len(batch_target_labels))
        preds = torch.tensor(list(itertools.chain.from_iterable(batch_labels))).cuda()
        target = torch.tensor(list(itertools.chain.from_iterable(batch_target_labels))).cuda()
        super(MyConfusionMatrix, self).update(preds, target)


class MyCM(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("ground_truths", default=[], dist_reduce_fx="sum")
        self.add_state("predictions", default=[], dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class MyConfusionMatrix1(ConfusionMatrix):
    @staticmethod
    def _get_highest_score_label(predicted_boxes, iou_scores):
        iou_filter = (iou_scores > 0.5).squeeze()
        predictions_per_ground_truth = predicted_boxes[iou_filter]
        # TODO: nondeterministic output of argmax for equal scores (rare?)
        if not predictions_per_ground_truth.numel(): return 0  # if predictions are empty return background class
        prediction_scores = predictions_per_ground_truth.narrow(-1, start=0, length=1)
        prediction_labels = predictions_per_ground_truth.narrow(-1, start=1, length=1).int()
        best_score_indice = prediction_scores.argmax()
        return prediction_labels[best_score_indice].item()

    def update(self, records, preds):
        batch_target_labels = []
        batch_labels = []
        # matching detections with targets
        for image_targets, image_preds in zip(records, preds):
            target_boxes = torch.Tensor([bbox.xyxy for bbox in image_targets['bboxes']])
            predicted_bboxes = torch.stack([
                torch.Tensor([score, label, *bbox.xyxy])  # bind scores, labels and boxes
                for score, label, bbox in zip(image_preds['scores'], image_preds['labels'], image_preds['bboxes'])
            ])
            iou_scores = box_iou(predicted_bboxes.narrow(-1, 2, 4), target_boxes)
            ious_per_ground_truth = iou_scores.split(split_size=1, dim=1)
            labels = [self._get_highest_score_label(predicted_bboxes, iou) for iou in ious_per_ground_truth]

            batch_labels.append(labels)
            batch_target_labels.append(image_targets['labels'])

        assert(len(batch_labels) == len(batch_target_labels))
        preds = torch.tensor(list(itertools.chain.from_iterable(batch_labels))).cuda()
        target = torch.tensor(list(itertools.chain.from_iterable(batch_target_labels))).cuda()
        super(MyConfusionMatrix, self).update(preds, target)