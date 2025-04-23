from datetime import datetime
from pathlib import Path

from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI

from base_model import LitBaseModel
from data_module import LitDataModule
from models import *  # Models need to be imported for the CLI to work


def main(args: ArgsType = None):
    save_dir = Path("./logs")
    name = datetime.now().strftime("%Y-%m-%d")
    version = datetime.now().strftime("%H-%M-%S")

    logger = {
        "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
        "init_args": {
            "save_dir": str(save_dir),
            "name": name,
            "version": version,
            "default_hp_metric": False,
        },
    }

    checkpointer = ModelCheckpoint(
        dirpath=str(save_dir / name / version / "checkpoints"),
        filename="epoch={epoch}-step={step}-val_loss={val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    cli = LightningCLI(
        LitBaseModel,
        LitDataModule,
        subclass_mode_model=True,
        args=args,
        trainer_defaults={"logger": logger, "callbacks": checkpointer},
    )


if __name__ == "__main__":
    main()
