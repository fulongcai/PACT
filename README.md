

# 🚁 Following the Flow: Advection-Consistent Modeling


<p align="center">
<img src="imgs\logo.png"  width='800' />
 </a>
</p>
The official implementation of "Following the Flow: Advection-Consistent Modeling for Event-based Small Object Detection"

## 🌟 Abstract

Event cameras enable high-frequency visual perception with microsecond latency, offering advantages for dynamic scenes. However, event-based small object detection remains challenging due to sparse asynchronous measurements and weak object responses that are easily disrupted by noise. Limited spatial support causes small-object signals to lose temporal continuity, resulting in fragmented and unstable predictions. To address this issue, we propose a physics-guided advection-consistent modeling framework, termed PACT, which formulates event evolution as a motion-driven feature transport process. Instead of relying solely on local spatio-temporal aggregation, PACT propagates features along estimated velocity fields and enforces trajectory-level consistency through advection constraints. This design preserves weak event responses over time and prevents their degradation under complex background interference. Technically, PACT integrates motion-aware feature extraction with a differentiable advection-based transport operator, enabling coherent motion representation and effective noise suppression during temporal evolution. Extensive experiments on benchmark event-based datasets demonstrate that PACT consistently outperforms state-of-the-art methods, achieving improvements of 20.72% in IoU and 15.03% in accuracy while maintaining comparable computational efficiency.

---

## 📊EV-UAV dataset
- ### Comparison between event camera and RGB camera
<p align="center">
<img src="imgs\left_top.jpg"  width='500' />
 </a>
</p>
The RGB camera can only capture the objects under normal light, while the event camera  can capture objects under various extreme lighting conditions. And the event camera can capture the continuous motion  trajectory of the small object (shown as the red curve).

- ### Comparison between EV-UAV and other event-based object detection datasets


<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Dataset</th>
    <th class="tg-c3ow" rowspan="2">#AGV.UAV scale</th>
    <th class="tg-c3ow" rowspan="2">Label Type</th>
    <th class="tg-c3ow" rowspan="2">UAV Sequence Ratio</th>
    <th class="tg-c3ow" rowspan="2">UAV centric</th>
    <th class="tg-c3ow" colspan="3">Lighting conditions</th>
    <th class="tg-0pky" colspan="2">Object</th>
    <th class="tg-0pky" rowspan="2">Year</th>
  </tr>
  <tr>
    <th class="tg-0pky">BL</th>
    <th class="tg-0pky">NL</th>
    <th class="tg-0pky">LL</th>
    <th class="tg-0pky">MS</th>
    <th class="tg-0pky">MT</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark">VisEvent</a></td>
    <td class="tg-0pky">84×66 pixels</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">15.97</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2023</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Event-AHU/EventVOT_Benchmark">EventVOT</a></td>
    <td class="tg-0pky">129×100 pixels</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">8.41</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Event-AHU/OpenEvDET">EvDET200K</a></td>
    <td class="tg-0pky">68×45 pixels</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">3.57</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://zenodo.org/records/10281437">F-UAV-D</a></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">100</td>
    <td class="tg-0pky"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/MagriniGabriele/NeRDD">NeRDD</a></td>
    <td class="tg-0pky">55×31 pixels</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">100</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky">EV-UAV</td>
    <td class="tg-0pky">6.8×5.4 pixels</td>
    <td class="tg-0pky">Seg</td>
    <td class="tg-0pky">100</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2025</td>
  </tr>
</tbody></table>
Currently, event-based object detection datasets  primarily focus on autonomous driving and general object detection.  There is limited attention given to  datasets that are  exclusively designed for UAV detection. We provide a comprehensive summary of existing datasets, highlighting the scarcity of event-based datasets for UAV object detection.

- ### Benchmark Features and Statistics
<img src="imgs\datasets.jpg" style="zoom: 25%;" />

EV-UAV contains 147 event sequences with **event-level annotations**, covering **challenging scenarios** like high-brightness and low-light conditions, with **targets** **averaging 1/50 the size in existing datasets**.

---

## 📂 Structure of EV-UAV

The file structure of the dataset is as follows:
```
EV-UAV/
├── test/          
│   ├── test_000.npz    
│   ├── test_001.npz
│   ├──.....
├── train          
├── val          

```

---

## 📝 Data Format

Event data is stored in `.npz` format, it contains three files (i.e., 'evs_norm', 'ev_loc' and 'ev').

**'ev'** is the raw event data.

- **x, y:** Pixel coordinates of the event.
- **timestamp:** Time of event occurrence (microseconds).
- **polarity:** Polarity of brightness change (+1 or -1).
- **label:** Indicates if it's the target (0 or 1).
- **id:** Identity of the target .

Example:

```
x    y   timestamp  polarity label id
100 200  1            1        0    0 
128 258  4000        -1        1    5
```



**'evs_norm'**  is the normalized event data.

Example:
```
x        y   timestamp  polarity label id
0.289 0.769  0            1        0    0 
0.369 0.992  0.5         -1        1    5
```



**'ev_loc'** is the coordinate of the event in point cloud space.

Example:
```
x    y   timestamp  
100 200  1           
128 258  4000      
```




## ⬇️ Dataset

The  EV-UAV dataset can be download from  [Baidu Netdisk](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2). Extracted code: sbr2 

---

# :triangular_flag_on_post:Baseline

Leveraging the spatiotemporal correlation characteristics of moving targets in event data, we propose EV-SpSegNet, a direct segmentation network for sparse event point clouds, and design a spatiotemporal correlation loss function that optimizes target event extraction by evaluating local spatiotemporal consistency.

### Event based Sparse Segmentation Network


Event based Sparse Segmentation Network (EV-SpSegNet) employs a U-shaped encoder-decoder architecture, integrating three key components: the GDSCA module (Grouped Dilated Sparse Convolution) for multi-scale temporal feature extraction, the Sp-SE module for feature fusion, and the Patch Attention block for voxel downsampling and global context modeling.

<img src="imgs\framework.png" width='900' />



- ### STCLOSS


We introduce a spatiotemporal correlation loss that encourages the network to retain more events with high spatiotemporal correlation while discarding more isolated noise.
<p align="center">
<img src="imgs\stcloss1.png"  width='300' />
 </a>
</p>
<p align="center">
<img src="imgs\stcloss2.png"  width='300' />
 </a>
</p>

# 🚀Installation

1) Create a new conda environment

```
conda create -n evuav python=3.8
conda activate evuav
```

2) Install dependencies

```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

3) Install  [spconv](https://github.com/traveller59/spconv)

4) Compile the external C++ and CUDA ops.

```
cd ev-spsegnet/lib/hais_ops
export CPLUS_INCLUDE_PATH={conda_env_path}/hais/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
```

## 🎯Running code

**1) Configuration file**: change the dataset root and the model save root by yourself

```python
cd configs/evisseg_evuav.yaml
```

**2) Training**

```python
train.py
```

**3) Testing**

```python
test.py
```
**4) Pre_trained weights**

The pre_trained weights can be download  [here](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4). Extracted code:ztp4


---

## Citation

If you use this work in your research, please cite it:

```bibtex
@misc{chen2025eventbasedtinyobjectdetection,
      title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline}, 
      author={Nuo Chen and Chao Xiao and Yimian Dai and Shiman He and Miao Li and Wei An},
      year={2025},
      eprint={2506.23575},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23575}, 
}
```
---
## Acknowledgement

The code is based on [HAIS](https://github.com/hustvl/HAIS) and [spconv](https://github.com/traveller59/spconv). 

