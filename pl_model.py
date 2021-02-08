from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from icevision import COCOMetric, ClassMap
from icevision.models import efficientdet
from .metrics import MyConfusionMatrix
import wandb
import pytorch_lightning as pl

__all__ = ['BaseModel', 'EffDetModel']


@dataclass
class TimmConfig:
    opt:str = 'fusedmomentum'
    weight_decay:float = 4e-5
    lr: float = 0.01
    momentum: float = 0.9

    opt_eps: float = 1e-3
    # opt_betas
    # opt_args

    epochs: int = 300
    lr_noise: tuple = (0.4, 0.9)
    sched: str = 'cosine'
    min_lr: float = 1e-5
    decay_rate: float = 0.1
    warmup_lr: float = 1e-4
    warmup_epochs: int = 5
    cooldown_epochs: int = 10

    lr_cycle_limit:int = 1
    lr_cycle_mul: float = 1.0
    lr_noise_pct: float = 0.67
    lr_noise_std: float = 1.0
    seed: int = 42


class BaseModel(efficientdet.lightning.ModelAdapter):
    def configure_optimizers(self):
        optimizer = create_optimizer(self.timm_config, self.model)
        lr_scheduler, num_epochs = create_scheduler(self.timm_config, optimizer)
        return [optimizer], [lr_scheduler]


class EffDetModel(BaseModel):
    def __init__(self, num_classes: int, img_size: int, model_name: Optional[str] = "tf_efficientdet_lite0", **timm_args):
        model = efficientdet.model(model_name=model_name, num_classes=num_classes, img_size=img_size, pretrained=True)
        # TODO: change this once pl-mAP is merged: https://github.com/PyTorchLightning/pytorch-lightning/pull/4564
        metrics = [COCOMetric(print_summary=True)]
        self.timm_config = TimmConfig(**timm_args)
        self.pl_metrics = [MyConfusionMatrix(num_classes=num_classes+1)]
        self.__num_val_dataloaders: int = 0
        super().__init__(model=model, metrics=metrics)

    def validation_step(self, batch, batch_idx, dataset_idx: int = 0):
        # execute validation on batch
        (xb, yb), records = batch

        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(raw_preds["detections"], 0)
            val_losses = {f'val_{key}': value for key, value in raw_preds.items() if 'loss' in key}
            loss = efficientdet.loss_fn(raw_preds, yb)

        for metric in self.metrics[dataset_idx]:
            metric.accumulate(records=records, preds=preds)

        # for metric in self.pl_metrics[dataset_idx]:
        #     metric.update(records=records, preds=preds)

        # loggin losses in step
        self.log_dict(val_losses, prog_bar=True)

    def validation_epoch_end(self, epoch_output):
        # deprecated?? trainer.evaluation_loop.py @ 210 - for pl factory Metrics in self.evaluation_callback_metrics
        # epoch_output is a list of step_outputs per dataloader shape: [0..n_dls, 0..n_batches]
        for dataset_idx in range(self.__num_val_dataloaders):
            self.finalize_metrics(dataset_idx)

        # have to log to wandb directly
        # class_names = self.get_dataset_class_map()._id2class
        # import numpy as np
        # self.logger.experiment.log({"conf_mat": wandb.plot.confusion_matrix(
        #     preds=np.array([1,2,1,4,5]), y_true=np.array([1,1,1,4,5]), class_names=class_names[:6])})
        # self.logger.experiment.log_artifact()

        # the len of this must be kept as len of dataloaders, otherwise pl ignores idx
        return epoch_output

    def test_step(self, batch, batch_idx, dataset_idx=0):
        return self.validation_step(batch, batch_idx, dataset_idx)

    def test_epoch_end(self, *args, **kwargs):
        self.validation_epoch_end(*args, **kwargs)

    def setup(self, stage_name):
        if isinstance(self.val_dataloader.dataloader, list):
            self.__num_val_dataloaders = len(self.val_dataloader.dataloader)
        else:
            self.__num_val_dataloaders = 1

        # create separate metrics for each dataloader
        self.metrics = [
            [deepcopy(metric) for metric in self.metrics]
            for _ in range(self.__num_val_dataloaders)
        ]
        self.pl_metrics = nn.ModuleList(
            [nn.ModuleList([deepcopy(pl_metric) for pl_metric in self.pl_metrics])
             for _ in range(self.__num_val_dataloaders)]
        )

        # if hasattr(self.test_dataloader, 'dataloader'):
        #     # create separate metrics for each dataloader
        #     # TODO fix for single test_dl
        #     self.metrics = [[deepcopy(metric) for metric in self.metrics] for dl in self.test_dataloader.dataloader]

    def finalize_metrics(self, dataset_idx: int = 0) -> None:
        # self.metric is a list of lists so we flatten it
        # for metric in itertools.chain(*self.metrics):
        for metric in self.metrics[dataset_idx]:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                # TODO: metric logging with forced dataset_idx
                self.log(f"{metric.name}/{k}/dl_idx_{dataset_idx}", v)

    def get_dataset_class_map(self):
        class_map: ClassMap
        class_map_valid = self.val_dataloader.dataloader.dataset.records[0].class_map
        class_map_train = self.train_dataloader.dataloader.dataset.records[0].class_map
        return class_map_valid

    def freeze_to_head(self, train_class_head=True, train_bbox_head=False):
        """
        Freezes the model up to the head part.
        Parameters control whether to train labels classifier and bbox regressor.
        """
        self.freeze()
        for param in self.model.model.box_net.parameters():
             param.requires_grad = train_bbox_head
        for param in self.model.model.class_net.parameters():
            param.requires_grad = train_class_head
        self.train()
