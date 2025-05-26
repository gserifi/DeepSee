import sys
from datetime import datetime
from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI

from base_model import LitBaseModel
from data_module import LitDataModule
from losses import *  # Losses need to be imported for the CLI to work
from models import *  # Models need to be imported for the CLI to work


class CustomCheckpoint(Callback):
    def __init__(self, dirpath: Path, every_n_epochs=5):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(
                str(self.dirpath / f"epoch={trainer.current_epoch + 1}.ckpt")
            )


def main(args: ArgsType = None):
    save_dir = Path("./logs")
    name = datetime.now().strftime("%Y-%m-%d")
    model_name = Path(
        sys.argv[[x in ["-c", "--config"] for x in sys.argv].index(True) + 1]
    ).stem
    version = f"{datetime.now().strftime('%H-%M-%S')}_{model_name}"

    logger = {
        "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
        "init_args": {
            "save_dir": str(save_dir),
            "name": name,
            "version": version,
            "default_hp_metric": False,
        },
    }

    # checkpointer = ModelCheckpoint(
    #     dirpath=str(save_dir / name / version / "checkpoints"),
    #     filename="epoch={epoch}-step={step}-val_loss={val/loss:.3f}",
    #     monitor="val/loss",
    #     mode="min",
    #     save_top_k=3,
    #     auto_insert_metric_name=False,
    # )

    checkpointer = CustomCheckpoint(
        dirpath=save_dir / name / version / "checkpoints",
    )

    cli = LightningCLI(
        LitBaseModel,
        LitDataModule,
        subclass_mode_model=True,
        args=args,
        trainer_defaults={
            "logger": logger,
            "callbacks": checkpointer,
            "enable_checkpointing": False,
        },
    )


if __name__ == "__main__":
    main()
