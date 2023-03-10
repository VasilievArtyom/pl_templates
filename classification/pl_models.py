import torch
from torch import optim
from ranger21 import Ranger21
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from attrdict import AttrDict
import math
from models.model import TorchModel
from losses.loss import TorchLoss
import os
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy



class TrainModel(pl.LightningModule):
    def __init__(
            self,
            config,
            train_loader,
            val_loader
    ):
        super(TrainModel, self).__init__()
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_training_steps = math.ceil(len(self.train_loader) / len(config['trainer']['devices']))
        self.model = TorchModel(**config['model'])
        if config['weights'] is not None:
            self.model.load_state_dict(torch.load(config['weights'], map_location='cpu'))
        self.criterion = TorchLoss()
        self.save_hyperparameters(AttrDict(config))

        # set train metrics
        self.train_acc = MulticlassAccuracy(num_classes=config['model']['output'])
        self.val_acc = MulticlassAccuracy(num_classes=config['model']['output'])


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.train_acc.update(
            y_hat.argmax(dim=1), y
        )
        self.log('loss/train', loss, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log('acc/train', self.train_acc.compute(), on_step=False, on_epoch=True, rank_zero_only=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.train_acc.reset()

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return {
            'loss': loss,
            'y': y,
            'y_hat': y_hat
        }

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def validation_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
            if key == "loss":
                out_val[key] = torch.stack([o[key] for o in outputs])
            else:
                out_val[key] = torch.cat([o[key] for o in outputs], dim=0)

        for key in out_val.keys():
            out_val[key] = self.sync_across_gpus(out_val[key])

        loss = out_val['loss'].mean()
        self.val_acc.update(
            out_val['y_hat'].argmax(dim=1), out_val['y']
        )

        self.log('loss/val', loss, prog_bar=False, rank_zero_only=True, sync_dist=False)
        self.log('acc/val', self.val_acc.compute(), sync_dist=False, rank_zero_only=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.hparams.optimizer == "ranger21":
            optimizer = Ranger21(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.hparams.lr,
                num_batches_per_epoch=self.num_training_steps,
                **self.config['optimizer_params']
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9, nesterov=True,
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hparams.optimizer}")

        scheduler_params = AttrDict(self.hparams.scheduler_params)
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params.patience,
                min_lr=scheduler_params.min_lr,
                factor=scheduler_params.factor,
                mode=scheduler_params.mode,
                verbose=scheduler_params.verbose,
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': scheduler_params.target_metric
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps * scheduler_params.warmup_epochs,
                num_training_steps=int(self.num_training_steps * self.config['trainer']['max_epochs'])
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            raise ValueError(f"Unknown sheduler name: {self.hparams.sheduler}")

        return [optimizer], [lr_scheduler]


class TestModel(pl.LightningModule):
    def __init__(
            self,
            config,
            test_loader,
            test_dataset
    ):
        super(TestModel, self).__init__()
        self.config = config
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.model = TorchModel(**config['model'])
        state_dict = torch.load(config['weights'], map_location='cpu')['state_dict']
        self.load_state_dict(state_dict, strict=True)
        self.save_hyperparameters(AttrDict(config))

    def test_dataloader(self):
        return self.test_loader

    def test_step(self, batch, batch_idx):
        x, ids = batch
        y_hat = self.model(x)
        y_hat = torch.argmax(y_hat, dim=1)
        return {
            "y_hat": y_hat,
            "idx": ids
        }

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([o['y_hat'] for o in outputs], dim=0).cpu().detach().tolist()
        ids = torch.cat([o['idx'] for o in outputs], dim=0).cpu().detach().tolist()
        data = [[self.test_dataset.images[idx], prediction] for idx, prediction in zip(ids, y_hat)]
        df = pd.DataFrame(data, columns=['filename', *self.hparams.classnames]).drop_duplicates()
        file_path = os.path.join(self.hparams.save_path, self.hparams.test_name, "predictions.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
