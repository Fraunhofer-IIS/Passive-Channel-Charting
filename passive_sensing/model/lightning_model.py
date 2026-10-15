import argparse
import lightning as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ChainedScheduler
from passive_sensing.lr_schedulers import LinearWarmupScheduler, ReduceLROnPlateauWithWarmup
from passive_sensing.model.transformer import FingerprintingTransformer
from passive_sensing.model.simple_cnn import Net as CNN
from passive_sensing.model.simple_cnn import CNNOptimized
from passive_sensing.model.resnet import ResNet

models = {
    "CNN": lambda in_dims: CNN(in_dims=in_dims),
    "CNNOptimized": lambda in_dims: CNNOptimized(in_dims=in_dims, out_dim=2),
    "ResNet": lambda in_dims: ResNet(in_dims=in_dims ,dims=[64,128,256], kernels=[3,3,3], strides=[2,2,2], num_blocks=[3,3,3], bias=True), 
    "TF" : lambda in_dims: FingerprintingTransformer(in_dims) 
}


class PositioningModel(L.LightningModule):


    def __init__(self, model_type: str, learning_rate: float = 1e-5, warmup_epochs: float = 10, in_channels: int = 1, scheduler_monitor_metric = "validation/loss", *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.model_type = model_type
        self.warmup_epochs = warmup_epochs
        self.scheduler_monitor_metric = scheduler_monitor_metric

        self.lr = learning_rate
        self.model = models.get(model_type)(in_channels)

    def forward(self, inputs):
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            # If the batch has only 2 elements, we have fingerprinting input
            inputs, labels = batch
            output = self(inputs)

            loss = torch.nn.functional.mse_loss(output, labels)

        elif len(batch) > 2:
            # If we have more than 2 inputs, we have channel charting inputs
            if len(batch) == 4:
                # Batch has weights
                input1, input2, target_distances, weight = batch
            elif len(batch) == 3:
                # Batch has no weights
                input1, input2, target_distances = batch
                weight = torch.ones_like(target_distances)

            out1 = self(input1)
            out2= self(input2)

            distances = out1 - out2
            distances = torch.linalg.norm(distances, dim=-1)

            loss = torch.nn.functional.mse_loss(distances, target_distances, reduction="none")
            loss = loss * weight
            loss = loss.mean()
        else:
            raise Exception("Unknown dataset format")
        
        self.log("training/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            inputs, labels = batch
            output = self(inputs)

            loss = torch.linalg.norm(output - labels, dim=1)
            self.log("validation/loss_ce90", torch.quantile(loss, .9))
            loss = loss.mean()

        elif len(batch) == 4:
            input1, input2, target_distances, weight = batch
            out1 = self(input1)
            out2= self(input2)

            distances = out1 - out2
            distances = torch.linalg.norm(distances, dim=-1)

            loss = torch.nn.functional.mse_loss(distances, target_distances, reduction="none")
            loss = loss * weight
            loss = loss.mean()
        else:
            raise Exception("Unknown dataset format")

        self.log("validation/loss", loss, sync_dist=True, prog_bar=True)

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        y = self(x)

        return y

    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            inputs, labels = batch
            output = self(inputs)

            loss = torch.linalg.norm(output - labels)

        elif len(batch) == 4:
            input1, input2, target_distances, weight = batch
            out1 = self(input1)
            out2= self(input2)

            distances = out1 - out2
            distances = torch.linalg.norm(distances, dim=-1)

            loss = torch.nn.functional.mse_loss(distances, target_distances, reduction="none")
            loss = loss * weight
            loss = loss.mean()
        else:
            raise Exception("Unknown dataset format")

        self.log("test/loss", loss, sync_dist=True)

        return loss
    

    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        reduce_lr = ReduceLROnPlateauWithWarmup(optimizer, mode='min', factor=0.3, min_lr=1e-5, patience=20, warmup_epochs=self.warmup_epochs, inital_lr=self.lr)
        reduce_lr = ReduceLROnPlateauWithWarmup(optimizer, mode='min', factor=0.5, min_lr=3e-5, patience=30, warmup_epochs=self.warmup_epochs, inital_lr=self.lr)
        return [optimizer], [{'scheduler': reduce_lr, 'monitor': self.scheduler_monitor_metric, 'interval': 'epoch'}]
        # return optimizer


    @staticmethod
    def add_cmd_args(argParser: argparse.ArgumentParser):
        argParser.add_argument("-r", "--learning-rate", default=1e-3, type=float, help="Learning rate")
        argParser.add_argument("-we", "--warmup-epochs", default=10, type=float, help="Number of epochs to apply a linear learning rate warmup")
        argParser.add_argument("-m", "--model-type", default="ResNet", required=True, choices=models.keys() ,type=str, help="Model archtiecture that is used for training")