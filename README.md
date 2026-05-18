# ST-GDance: Long-Term and Collision-Free Group Choreography from Music
<h3>Jing Xu, Weiqiang Wang, Cunjian Chen, Jun Liu, Qiuhong Ke</h3>

Official implementation of ST-GDance
📢 Accepted at **BMVC 2025 (Oral)**

[[arXiv](https://arxiv.org/abs/2507.21518)] [[Project Page](https://yilliajing.github.io/ST-GDance-Website/)]

## 1. Environment Setup

First of all, please setup the virtual environment with Anaconda:

```bash
conda create -n stgdance python=3.7.12
conda activate stgdance
conda install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Data

We use the **[GDANCE](https://github.com/aioz-ai/AIOZ-GDANCE)** dataset to train and evaluate our models. Please **[download](https://huggingface.co/datasets/aiozai/AIOZ-GDANCE)** and extract the data into `./data/gdance/`. If you extract it to a different location, you will need to update the data path in the configuration accordingly.

Our model takes music features as input to generate corresponding dance motions. We provide pre-extracted **[Jukebox features](https://huggingface.co/aiozai/JukeBoxFeatures/resolve/main/jukebox_features.zip)** derived from the GDANCE music sequences for your convenience.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{Xu_2025_BMVC,
  author    = {Jing Xu and Weiqiang Wang and Cunjian Chen and Jun Liu and Qiuhong Ke},
  title     = {ST-GDance: Long-Term and Collision-Free Group Choreography from Music},
  booktitle = {36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025},
  publisher = {BMVA},
  year      = {2025},
  url       = {https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_66/paper.pdf}
}
```
