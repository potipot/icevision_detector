import itertools
from typing import Dict, List, Tuple

import sklearn
import torch
import numpy as np

from icevision import DetectedBBox
from pytorch_lightning.metrics import ConfusionMatrix, Metric
from pytorch_lightning.loggers import wandb as pl_wandb
from torchvision.ops import box_iou
from icevision.metrics.metric import Metric as IceMetric

import wandb


def zeroify_items_below_threshold(iou_scores:torch.Tensor, threshold:float) -> torch.Tensor:
    return iou_scores*(iou_scores>threshold).byte()


def couple_with_targets(predicted_bboxes, iou_scores) -> List:
    """Connects detected bounding boxes with ground truths by iou > 0"""
    ious_per_target = iou_scores.split(1, dim=1)
    return [list(itertools.compress(predicted_bboxes, iou.bool())) for iou in ious_per_target]


def pick_best_score_labels(predicted_bboxes, confidence_threshold: float = 0.5):
    # fill with dummy if list of predicted labels is empty
    BACKGROUND_IDX = 0
    dummy = DetectedBBox(0, 0, 0, 0, score=1.0, label=BACKGROUND_IDX)
    best_labels = []
    # pick the label that fits best given ground truth
    for ground_truth_predictions in predicted_bboxes:
        ground_truth_predictions = [
            prediction if prediction.score > confidence_threshold else dummy
            for prediction in ground_truth_predictions
        ]
        best_prediction = max(ground_truth_predictions, key=lambda x: x.score, default=dummy)
        best_labels.append(best_prediction.label)
    return best_labels


def pairwise_iou(predicted_bboxes, target_bboxes):
    stacked_preds = [bbox.to_tensor() for bbox in predicted_bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target_bboxes]
    stacked_targets = torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    return box_iou(stacked_preds, stacked_targets)


class SimpleConfusionMatrix(IceMetric):
    def __init__(self, confidence_threshold: float = 0.001, iou_threshold: float = 0.5):
        super(SimpleConfusionMatrix, self).__init__()
        self.ground_truths = []
        self.predictions = []
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self.labels = None
        self.cm_display = None

    def _reset(self):
        self.ground_truths = []
        self.predictions = []
        self.labels = None

    def accumulate(self, records, preds):
        if self.labels is None: self.labels = next(iter(records))['class_map']._id2class
        for image_targets, image_preds in zip(records, preds):
            target_bboxes = image_targets['bboxes']
            target_labels = image_targets['labels']
            # skip if empty ground_truths
            if not target_bboxes: continue
            predicted_bboxes = [
                DetectedBBox(*bbox.xyxy, score=score, label=label)
                for bbox, score, label in zip(image_preds['bboxes'], image_preds['scores'], image_preds['labels'])
            ]
            # get torchvision iou scores (requires conversion to tensors)
            iou_scores = pairwise_iou(predicted_bboxes, target_bboxes)
            # TODO: see what happens if that_match is empty
            that_match = torch.any(iou_scores > self._iou_threshold, dim=1)
            iou_scores = iou_scores[that_match]
            iou_scores = zeroify_items_below_threshold(iou_scores, threshold=self._iou_threshold)

            # need to use compress cause list indexing with boolean tensor isn't supported
            predicted_bboxes = list(itertools.compress(predicted_bboxes, that_match))
            predicted_bboxes = couple_with_targets(predicted_bboxes, iou_scores)
            predicted_labels = pick_best_score_labels(predicted_bboxes, confidence_threshold=self._confidence_threshold)

            assert(len(predicted_labels) == len(target_labels))
            self.ground_truths.extend(target_labels)
            self.predictions.extend(predicted_labels)

    def finalize(self):
        assert(len(self.ground_truths) == len(self.predictions))
        self.ground_truths = np.array(self.ground_truths)
        self.predictions = np.array(self.predictions)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=self.ground_truths, y_pred=self.predictions)
        self.cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=self.labels)
        return {'dummy_value_for_fastai': -1}

    def plot(self, **display_args):
        """
        A handle to plot the matrix in a jupyter notebook, potentially this could also be passed to save_fig
        """
        return self.cm_display.plot(**display_args).figure_

    def log(self, logger_object) -> None:
        if isinstance(logger_object, pl_wandb.WandbLogger):
            # wandb.sklearn.plot_confusion_matrix(y_true=self.ground_truths, y_pred=self.predictions, labels=self.labels)
            logger_object.experiment.log({
                "conf_mat": wandb.plot.confusion_matrix(
                    preds=self.predictions, y_true=self.ground_truths, class_names=self.labels)
            })
        self._reset()
        return


