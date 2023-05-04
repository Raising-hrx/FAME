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
Download EntailmentBankQA dataset.


# Evaluation

Download our trained models for direct reproduction, including xxx. Unzip the files and place them in `exp/` folder. Run the following commands to reproduce the results.

## EntailmentBankQA

```
sh scripts/eval_scripts/reason_EBQA.sh
```

## EntailmentBank Task3

```
sh scripts/eval_scripts/reason_EB.sh
```



# Training

## Controller
```
sh scripts/training_scripts/train_Controller_ddp.sh
```

## Entailment Module
Please refer to [MetGen](https://github.com/Raising-hrx/MetGen) for the training of the single-step entialment module.

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
comming soon.
```