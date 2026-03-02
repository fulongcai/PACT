# 🚁 Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection

The official implementation of **"Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection"**.

## 🌟 Abstract

Event cameras enable high-frequency visual perception with microsecond latency, which makes them well suited for dynamic scenes. However, event-based small object detection is still difficult because event measurements are sparse and asynchronous, while target responses are weak and easily corrupted by noise. Limited spatial support further breaks temporal continuity, which leads to fragmented and unstable predictions. To address this problem, we propose a physics-guided advection-consistent modeling framework, termed **PACT**, which formulates event evolution as a motion-driven feature transport process. Instead of depending only on local spatio-temporal aggregation, PACT propagates features along estimated velocity fields and enforces trajectory-level consistency through advection constraints. This design preserves weak event responses over time and reduces their degradation under complex background interference. Technically, PACT combines motion-aware feature extraction with a differentiable advection-based transport operator, enabling coherent motion representation and effective noise suppression during temporal evolution. Extensive experiments on benchmark event-based datasets show that PACT consistently outperforms previous methods, achieving improvements of **20.72% in IoU** and **15.03% in accuracy** while maintaining comparable computational efficiency.

---

## 📊 Dataset

PACT is evaluated on the **EV-UAV** dataset.

EV-UAV is a benchmark designed for event-based UAV small object detection. It contains challenging scenes with complex backgrounds and adverse lighting conditions, which makes it suitable for evaluating temporal consistency and noise robustness in event-based perception.

Please organize the dataset in the following structure:

```text
EV-UAV/
├── train/
│   ├── train_000.npz
│   ├── train_001.npz
│   └── ...
├── val/
│   ├── val_000.npz
│   ├── val_001.npz
│   └── ...
└── test/
    ├── test_000.npz
    ├── test_001.npz
    └── ...
```

Each sample is stored in `.npz` format. The exact data definition follows the official EV-UAV release. In general, each file may include fields such as:

- `ev`: raw event stream
- `evs_norm`: normalized event representation
- `ev_loc`: event coordinates in point cloud space

For detailed dataset description, annotations, and download links, please refer to the official EV-UAV repository.

---

## 📝 Data Format

PACT follows the original EV-UAV data format.

**`ev`** contains the raw event stream.

- **x, y:** pixel coordinates of the event
- **timestamp:** event timestamp
- **polarity:** polarity of brightness change
- **label:** whether the event belongs to the target
- **id:** target identity

Example:

```text
x    y    timestamp  polarity  label  id
100  200  1          1         0      0
128  258  4000       -1        1      5
```

**`evs_norm`** contains the normalized event representation.

Example:

```text
x      y      timestamp  polarity  label  id
0.289  0.769  0          1         0      0
0.369  0.992  0.5        -1        1      5
```

**`ev_loc`** stores the event coordinates in point cloud space.

Example:

```text
x    y    timestamp
100  200  1
128  258  4000
```

---

# :triangular_flag_on_post: PACT Framework

PACT is a physics-guided framework for event-based small object detection. It models event evolution as a motion-driven transport process and improves temporal coherence by propagating features along estimated velocity fields. This design helps preserve weak target responses and suppresses background noise during temporal evolution.

### Framework Overview

The overall framework of PACT is shown below.

<p align="center">
<img src="imgs/overall.png" width="900" />
</p>

### Qualitative Results

Some qualitative visualization results are shown below.

<p align="center">
<img src="imgs/results.png" width="900" />
</p>

---

# 🚀 Installation

### 1) Create a new conda environment

```bash
conda create -n pact python=3.8
conda activate pact
```

### 2) Install dependencies

```bash
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### 3) Install [spconv](https://github.com/traveller59/spconv)

Please install `spconv` according to your local CUDA and PyTorch environment.

### 4) Compile the external C++ and CUDA ops

```bash
cd lib/hais_ops
export CPLUS_INCLUDE_PATH={conda_env_path}/hais/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
cd ../..
```

---

## 🎯 Running Code

**1) Configuration file**

Please modify the dataset root and checkpoint save path in the config file:

```bash
configs/evisseg_evuav.yaml
```

**2) Training**

```bash
python train.py
```

**3) Testing**

```bash
python test.py
```

**4) Pretrained weights**

Pretrained weights will be released **soon**.

---

## Citation

If you find this project useful in your research, please cite:

```bibtex
@article{pact_placeholder_2026,
  title   = {Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection},
  author  = {Author list coming soon},
  journal = {Conference / Journal coming soon},
  year    = {2026}
}
```

---

## Acknowledgement

This project is built upon **EV-SpSegNet**, [HAIS](https://github.com/hustvl/HAIS), and [spconv](https://github.com/traveller59/spconv). We sincerely thank the authors of these open-source projects for making their code publicly available.
