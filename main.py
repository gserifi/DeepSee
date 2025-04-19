from lightning.pytorch.cli import ArgsType, LightningCLI
from datetime import datetime

from base_model import LitBaseModel
from data_module import LitDataModule
from models import *  # Models need to be imported for the CLI to work


def main(args: ArgsType = None):
    logger = {
        "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
        "init_args": {
            "save_dir": "./logs",
            "name": f"{datetime.now().strftime('%Y-%m-%d')}",
            "version": f"{datetime.now().strftime('%H-%M-%S')}",
            "default_hp_metric": False,
        },
    }

    cli = LightningCLI(
        LitBaseModel,
        LitDataModule,
        subclass_mode_model=True,
        args=args,
        trainer_defaults={"logger": logger},
    )


if __name__ == "__main__":
    main()
