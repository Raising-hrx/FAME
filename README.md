# FAME
An implementation for "Faithful Question Answering with Monte-Carlo Planning"

<p align="center">
<img src="imgs/intro.jpg" alt="FAME" width="500"/>
</p>



# Requirements
- Python 3.8
- Ubuntu 22.04
- Python Packages

```
conda create -n fame python=3.8
conda activate fame
pip install -r requirements.txt
```

# Data
Download [Data](https://cloud.tsinghua.edu.cn/f/e83ffd99d1d2476da013/?dl=1) folder. 
The Data folder includes the EntailmentBank, EntailmentBankQA, training data for the controller, and training data for the verifier.
See `Data/Readme.md` for details.


# Evaluation

Download our trained models for direct reproduction, including 
[Controller](https://cloud.tsinghua.edu.cn/d/eb7a4bd23b064032b81d/),
[Entailment Module](https://cloud.tsinghua.edu.cn/f/3cba574163ee46c09ff1/?dl=1),
[Retriever](https://cloud.tsinghua.edu.cn/f/1cb9b5b705bf47dfb7dc/?dl=1),
and [Verifier](https://cloud.tsinghua.edu.cn/f/d16a42ce563a4cd68ff3/?dl=1).
Unzip the files and place them in `exp/` folder. Run the following commands to reproduce the results.

## EntailmentBankQA

```
sh scripts/eval_scripts/reason_EBQA.sh
```
The result will be saved in the `save_dir`.

## EntailmentBank Task3

```
sh scripts/eval_scripts/reason_EB.sh
```
The result will be saved in the `save_dir`.
Use the [offical evaluation code of EntailmentBank](https://github.com/allenai/entailment_bank) to evaluate automatically.


# Training


## Entailment Module
Please refer to [MetGen](https://github.com/Raising-hrx/MetGen) for the training of the single-step entialment module.

## Controller
```
sh scripts/training_scripts/train_Controller_ddp.sh
```

## Retriever
```
sh scripts/training_scripts/train_Retriever.sh
```

## Verifier
```
sh scripts/training_scripts/train_StepScorer.sh
```

# Citation
```
@inproceedings{hong2023fame,
  title={Faithful Question Answering with Monte-Carlo Planning},
  author={Ruixin Hong, Hongming Zhang, Hong Zhao, Dong Yu and Changshui Zhang},
  booktitle={The 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2023}
}
```