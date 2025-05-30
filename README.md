# DeepSee: Monocular Depth Estimation

<div align="center">
Aaron Zeller
&nbsp;&nbsp;&nbsp;&nbsp;
Gent Serifi
&nbsp;&nbsp;&nbsp;&nbsp;
Nicola Studer
&nbsp;&nbsp;&nbsp;&nbsp;

ETH Zurich, Switzerland

Computational Intelligence Lab
&#8226;
[Kaggle Competition](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025)
</div>

## Get Started

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install PyTorch

```bash
pip install torch torchvision
```

### Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

When adding dependencies, run the following command afterwards:

```bash
pip list --format=freeze > requirements.txt
```

### Setup Pre-Commit hooks (Development only)

```bash
pre-commit install
```

This will run `isort` and `black` for import sorting and code formatting to ensure consistency. Note that if the check
fails, the commit will be rejected. You can also run `isort .; black .` prior to committing to bring the code into the
right shape.

### Download Dataset

And put it inside `./data`.

### Train

```bash
python3 main.py fit -c configs/<config>.yaml
```

The `configs` folder contains various configurations of our
model. To run our final best-performing model, use the
`configs/best_model.yaml` file.

Note that this script will install Dinov2 checkpoints
from HuggingFace. This may take a while when running the script
for the first time.

Checkpoints and logs will be saved in the ``logs`` folder.

Monitor using TensorBoard:

```bash
tensorboard --logdir logs
```

### Test

```bash
python3 main.py test -c configs/<config>.yaml --ckpt_path <path_to_checkpoint>
```

The checkpoint path has to point to a valid checkpoint file (has to have been trained with the same config YAML file)
in the `logs` folder. The format looks like this: `logs/<yyyy-mm-dd>/<hh-mm-ss>/checkpoints/<ckpt_name>.ckpt`

This script evaluates the model our own custom holdout test set and prints the respective metrics.

### Inference

```bash
python3 main.py predict -c configs/<config>.yaml --ckpt_path <path_to_checkpoint>
python3 data/create_prediction_csv.py
```

These scripts will run the model on the Kaggle test set and create the CSV file.