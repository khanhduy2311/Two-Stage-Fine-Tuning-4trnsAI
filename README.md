# Two-stage-fine-tuning
This is the official code for the "Two-stage fine-tuning with ChatGPT data augmentation for learning class-imbalanced data". The paper can be found here:

https://www.sciencedirect.com/science/article/pii/S0925231224005721

## Install
Download the repo and unzip
```
wget https://github.com/TahaAslani/two-stage-fine-tuning/archive/refs/heads/main.zip
unzip main.zip
```

Install dependencies
```
pip install torch
pip install transformers
```
The codes were tested with transformers version 4.13.0 and torch version 1.8.1 with a compatible cuda.

## Prepare data
Download and unzip the SST-2 data from GLUE
```
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
```
Downsample data
```
python two-stage-fine-tuning-main/down_sample.py -i SST-2 -o down_sampled
```

## Run the experiments

### Run Two-stage Fine-tuning with chatGPT augmented data
#### Get the augmented data
First, download the augmented data and put it in the "down_sampled/0.2" folder
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PyTdS9Ev_OhsU2WQSQWRxBw8TV8Z27tB' -O down_sampled/0.2/aug.csv
```
Alternatively, you can manually download the augmented data from the link below and put it in the "down_sampled/0.2" folder

https://drive.google.com/file/d/1PyTdS9Ev_OhsU2WQSQWRxBw8TV8Z27tB/view?usp=sharing

#### Generating augmented data using ChatGPT (skip to Running the pipeline if you already downloaded the augmented data)
If you want to generate the augmented data, you can use the following command. You need to provide your OpenAI API Key as a text string
```
pip install openai
python two-stage-fine-tuning-main/gen_aug.py --data-path down_sampled/0.2 --api-key $OPEN_AI_API_KEY
```
Where $OPEN_AI_API_KEY is your OpenAI APKI key for charging.

Note that generating augmented data will take a long time. Moreover, due to the stochastic nature of chatGPT, the generated data will not be the same as the data used in the paper. As a consequence, the final results will change slightly. Skip this step if you already downloaded the augmented data.

#### Running the pipeline
After obtaining the augmented data, run the experiment
```
python two-stage-fine-tuning-main/two_stage_aug.py --data-path down_sampled/0.2 --output-path results/Two-stage-chatGPT --epoch-stage-1 1 --epoch-stage-2 1
```

### Run Two-stage Fine-tuning with reweighting
```
python two-stage-fine-tuning-main/two_stage_reweight.py --data-path down_sampled/0.2 --output-path results/Two-stage-reweight --epoch-stage-1 1 --epoch-stage-2 1
```

### Run Vanilla Fine-tuning
```
python two-stage-fine-tuning-main/two_stage_reweight.py --data-path down_sampled/0.2 --output-path results/Vanilla --epoch-stage-1 0 --epoch-stage-2 1
```

The results of each experiment will be saved in CSV in the corresponding folders.

## Cite as:
```
@article{VALIZADEHASLANI2024127801,
title = {Two-stage fine-tuning with ChatGPT data augmentation for learning class-imbalanced data},
journal = {Neurocomputing},
volume = {592},
pages = {127801},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127801},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224005721},
author = {Taha ValizadehAslani and Yiwen Shi and Jing Wang and Ping Ren and Yi Zhang and Meng Hu and Liang Zhao and Hualou Liang},
keywords = {Imbalanced data. Machine learning. Natural language processing. Data augmentation. Reweighting. BERT}
}
```
