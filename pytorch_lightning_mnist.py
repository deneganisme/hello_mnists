import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import random_split
import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # (log keyword is optional)
        # return {'loss': loss, 'log': {'train_loss': loss}}

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        result = pl.EvalResult()
        result.log('val_loss', loss)

        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        result = pl.EvalResult()
        result.log('val_loss', loss)

        return result


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--auto_lr', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--backend', type=str, default='ddp')

    args = parser.parse_args()

    logger.info("Creating train/val datasets...")
    train_dataset = MNIST(os.getcwd(),
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())

    n_train = int(len(train_dataset) * 0.9)
    n_val = int(len(train_dataset) * 0.1)

    assert n_train + n_val == len(train_dataset), \
        f"Mismatch: {n_train} + {n_val} != {len(train_dataset)}"

    train_set, val_set = random_split(train_dataset, lengths=[n_train, n_val])

    train_loader = DataLoader(train_set)
    val_loader = DataLoader(val_set)


    # Create test dataset
    logger.info("Done! Creating test dataset...")
    test_dataset = MNIST(os.getcwd(),
                         train=False,
                         download=True,
                         transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset)

    logger.info(f"Done!"
                f"\n# of train examples: {n_train}"
                f"\n# of val examples: {n_val}"
                f"\n# of test examples: {len(test_dataset)}")

    # init model
    model = LitModel(args)

    if args.patience is not None:
        early_stop_ckpt = EarlyStopping(monitor='val_loss',
                                        verbose=True,
                                        patience=args.patience)
    else:
        early_stop_ckpt = None

    mdl_ckpt = ModelCheckpoint(save_top_k=1, verbose=True, monitor='val_loss')

    profiler = SimpleProfiler()

    lightning_log_pth = '/lightning_logs'

    if not os.path.isdir(lightning_log_pth):
        logger.warning(f"Unable to find {lightning_log_pth} to log to! "
                       f"If not running Grid then ignore.")
        save_dir = ''
    else:
        save_dir = lightning_log_pth

    tensorboard = TensorBoardLogger(save_dir=save_dir)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=args.gpus if torch.cuda.is_available() else None,
                         auto_lr_find=bool(args.auto_lr),
                         logger=tensorboard,
                         checkpoint_callback=mdl_ckpt,
                         early_stop_callback=early_stop_ckpt,
                         distributed_backend=args.backend if torch.cuda.is_available() else None,
                         max_epochs=args.max_epochs)

    logger.info("Beginning training...")
    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)

    logger.info("Done! Beginning testing...")
    trainer.test(model, test_loader)
    logger.info("Done - job complete!")
