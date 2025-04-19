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

### Download Dataset

And put it inside `./data`.

### Train

```bash
python3 main.py fit -c configs/example.yaml
```

Monitor using TensorBoard:

```bash
tensorboard --logdir logs
```

### Test

TODO: I'll check how to do this properly.

```bash
python3 main.py test -c configs/example.yaml
```