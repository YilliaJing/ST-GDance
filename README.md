# ST-GDance

Official implementation of  
**"ST-GDance: Long-Term and Collision-Free Group Choreography from Music"**  
📢 Accepted at **BMVC 2025 (Oral)**

## 1. Environment Setup

First of all, please setup the virtual environment with Anaconda:

```bash
conda create -n stgdance python=3.7.12
conda activate stgdance
conda install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Data

We use the **[GDANCE](https://github.com/aioz-ai/AIOZ-GDANCE)** dataset to train and evaluate our models. Please **[download](https://huggingface.co/datasets/aiozai/AIOZ-GDANCE)** and extract the data into `./datasets/`. If you extract it to a different location, you will need to update the data path in the configuration accordingly.

Our model takes music features as input to generate corresponding dance motions. We provide pre-extracted **[Jukebox features](https://huggingface.co/aiozai/JukeBoxFeatures/resolve/main/jukebox_features.zip)** derived from the GDANCE music sequences for your convenience.
