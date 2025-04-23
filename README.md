# DeepSee: Monocular Depth Estimation

<div align="center">
Gent Serifi
&nbsp;&nbsp;&nbsp;&nbsp;
Nicola Studer
&nbsp;&nbsp;&nbsp;&nbsp;
Aaron Zeller

ETH Zürich

Computational Intelligence Lab
&#8226;
[Kaggle Competition](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025)
</div>

## Roadmap

- [ ] Plan Project

## Get Started

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

When adding dependencies, run the following command afterwards:

```bash
pip list --format=freeze > requirements.txt
```

### Setup Pre-Commit hooks

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

Monitor using TensorBoard:

```bash
tensorboard --logdir logs
```

### Test

```bash
python3 main.py test -c configs/<config>.yaml --ckpt_path <path_to_checkpoint>
```

### Inference

```bash
python3 main.py predict -c configs/<config>.yaml --ckpt_path <path_to_checkpoint>
python3 data/create_prediction_csv.py
```