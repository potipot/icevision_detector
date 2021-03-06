from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Any

import torch
from torch import nn
import pytorch_lightning as pl
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from icevision import COCOMetric#, SimpleConfusionMatrix
from icevision.models import efficientdet

__all__ = ['BaseModel', 'EffDetModel']

from icevision.models.ross.efficientdet import EfficientDetBackboneConfig


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


class BaseModel(pl.LightningModule):
    def __init__(self, model: nn.Module, metrics: List[Any] = None, **timm_args):
        super(BaseModel, self).__init__(model=model, metrics=metrics)
        self.n_train_dls, self.n_test_dls, self.n_valid_dls = None, None, None
        self.timm_config = TimmConfig(**timm_args)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.timm_config, self.model)
        lr_scheduler, num_epochs = create_scheduler(self.timm_config, optimizer)
        return [optimizer], [lr_scheduler]

    def setup(self, stage_name):
        def _get_num_of_dataloaders(dataloader):
            n: int
            dl = getattr(self, dataloader)
            assert(callable(dl))
            result = dl()
            if isinstance(result, list): n = len(self.val_dataloader.dataloader)
            elif isinstance(result, torch.utils.data.dataloader.DataLoader): n = 1
            elif result is None: n = 0
            else: raise RuntimeError
            return n
        self.n_train_dls = _get_num_of_dataloaders('train_dataloader')
        self.n_valid_dls = _get_num_of_dataloaders('val_dataloader')
        self.n_test_dls = _get_num_of_dataloaders('test_dataloader')

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


class EffDetModel(BaseModel, efficientdet.lightning.ModelAdapter):
    def __init__(self, num_classes: int, img_size: int, model_name: Optional[str] = "tf_efficientdet_lite0", **timm_args):
        backbone_config = EfficientDetBackboneConfig(model_name=model_name)
        model = efficientdet.model(
            backbone=backbone_config(pretrained=True),
            num_classes=num_classes,
            img_size=img_size
        )
        # TODO: change this once pl-mAP is merged: https://github.com/PyTorchLightning/pytorch-lightning/pull/4564
        metrics = [COCOMetric(print_summary=True)]#, SimpleConfusionMatrix()]
        super().__init__(model=model, metrics=metrics, **timm_args)

    def validation_step(self, batch, batch_idx, dataset_idx: int = 0):
        # execute validation on batch
        (xb, yb), records = batch

        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                raw_preds=raw_preds["detections"],
                records=records,
                detection_threshold=0
            )

            val_losses = {f'val_{key}': value for key, value in raw_preds.items() if 'loss' in key}
            loss = efficientdet.loss_fn(raw_preds, yb)

        for metric in self.metrics[dataset_idx]:
            # TODO: update old metric.accumulate(records=records, preds=preds)
            metric.accumulate(preds=preds)

        # logging losses in step
        self.log_dict(val_losses)

    def validation_epoch_end(self, epoch_output):
        # deprecated?? trainer.evaluation_loop.py @ 210 - for pl factory Metrics in self.evaluation_callback_metrics
        # epoch_output is a list of step_outputs per dataloader shape: [0..n_dls, 0..n_batches]
        # logging metrics in epoch_end
        for dataset_idx in range(len(self.metrics)):
            self.finalize_metrics(dataset_idx)

    def test_step(self, batch, batch_idx, dataset_idx=0):
        return self.validation_step(batch, batch_idx, dataset_idx)

    def test_epoch_end(self, *args, **kwargs):
        self.validation_epoch_end(*args, **kwargs)

    def setup(self, stage_name):
        super(EffDetModel, self).setup(stage_name=stage_name)

        # create separate metrics for each dataloader
        self.metrics = [
            [deepcopy(metric) for metric in self.metrics]
            for _ in range(self.n_valid_dls if stage_name == 'fit' else self.n_test_dls)
        ]
        # self.pl_metrics = nn.ModuleList(
        #     [nn.ModuleList([deepcopy(pl_metric) for pl_metric in self.pl_metrics])
        #      for _ in range(self.n_valid_dls)]
        # )

    def finalize_metrics(self, dataset_idx: int = 0) -> None:
        for metric in self.metrics[dataset_idx]:
            metric_logs = metric.finalize()
            # FIXME: disable logging in sanity check
            if self.trainer.running_sanity_check: return
            log = getattr(metric, 'log', None)
            if callable(log):
                log(self.logger)
            else:
                for k, v in metric_logs.items():
                    # TODO: metric logging with forced dataset_idx
                    self.log(f"{metric.name}/{k}/dl_idx_{dataset_idx}", v)

    def load_matching(self, checkpoint_path: str):
        this_model = self.model.state_dict()
        trained_model = torch.load(checkpoint_path)['state_dict']

        for (this_module, this_param), (loaded_module, loaded_param) in zip(this_model.items(), trained_model.items()):
            assert ('model.' + this_module == loaded_module), f'Models differ: {this_module}, {loaded_module}'
            if this_param.shape == loaded_param.shape:
                this_model[this_module] = loaded_param
            else:
                print(f'Weights not loaded: {this_module}: {this_param.shape=}, {loaded_param.shape=}')

        return self.model.load_state_dict(this_model)
