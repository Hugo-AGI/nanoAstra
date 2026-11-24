<div align="center">

# nanoAstra

### A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2611.29338-b31b1b.svg)](https://arxiv.org/abs/2611.29338)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2611.29338)
[![Project Page](https://img.shields.io/badge/Project-Page-1f4e8c.svg)](https://nanoastra.github.io)

Meng-Hao Guo<sup>1,\*,†</sup>, Yi Zhang<sup>1,\*</sup>, Zhe-Han Mo<sup>1,\*</sup>, Han-Zheng Wang<sup>1,\*</sup>, Jia-Jun Wang<sup>1,\*</sup>,
Xin-Sheng Chen<sup>1,\*</sup>, Jiaxi Song<sup>1</sup>, Yi-Xuan Deng<sup>1</sup>, Jia-Ning Li<sup>1</sup>, Xuanyu Chu<sup>1</sup>,
Hengrui Zhang<sup>1</sup>, Huan-ang Gao<sup>1</sup>, Yongming Rao<sup>2</sup>, Shi-Min Hu<sup>1,✉</sup>

<sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>2</sup>Tencent Hunyuan

<sup>\*</sup>Core Contributors &nbsp;&nbsp; <sup>†</sup>Project Lead &nbsp;&nbsp; <sup>✉</sup>Corresponding Author

</div>

<p align="center">
  <img src="https://nanoastra.github.io/static/images/teaser.jpg" width="560" alt="Comparison of VLA, WAM, and nanoAstra">
</p>

## Overview

**nanoAstra** trains a vision-language model (VLM) for generalizable robot manipulation through its native text interface, with no separate action head. At each decision point, the model writes the task **progress** and the next **semantic steps**, built from reusable, parameterized action primitives such as `move x +5.0` or `gripper open`.

- **Semantic action modeling.** More than 5,000 hours of heterogeneous robot demonstrations are filtered, geometrically standardized, and segmented at behavioral decision points into semantic steps with dense progress descriptions.
- **Capability-preserving SFT (CP-SFT).** A robot-specific system prompt softly confines manipulation outputs to a structured response space. Co-training with general VQA data helps keep the model's broader vision-language capabilities.
- **Zero-shot manipulation.** On RoboTwin 2.0's randomized benchmark, nanoAstra-9B reaches a **17.8%** zero-shot success rate, ahead of the evaluated VLA, WAM, and VLM baselines.
- **Streaming execution.** On a single H20 GPU, inference is faster than the motion it produces. Overlapping the two reduces the robot's idle time from **35.0%** to **6.9%**.

## News

- **2026-11** — Paper released on [arXiv](https://arxiv.org/abs/2611.29338).

## Code

Code will be publicly available in this repository. Stay tuned.

## Citation

If you find this work useful, please cite:

```bibtex
@article{guo2026glimpse,
  title   = {A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation},
  author  = {Guo, Meng-Hao and Zhang, Yi and Mo, Zhe-Han and Wang, Han-Zheng and
             Wang, Jia-Jun and Chen, Xin-Sheng and Song, Jiaxi and Deng, Yi-Xuan and
             Li, Jia-Ning and Chu, Xuanyu and Zhang, Hengrui and Gao, Huan-ang and
             Rao, Yongming and Hu, Shi-Min},
  journal = {arXiv preprint arXiv:2611.29338},
  year    = {2026}
}
```
