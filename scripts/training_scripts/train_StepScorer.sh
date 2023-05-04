cd <base_dir>/code

data_dir=<base_dir>/data/TeachableEntailmentWriterData/processed_step_data
code_dir=<base_dir>/code


model=microsoft/deberta-v3-large
CUDA_VISIBLE_DEVICES=0 python StepScorer.py \
--model_name_or_path $model --bs 16 \
--train_data $data_dir/step.train.v1.jsonl \
--dev_data $data_dir/step.dev.v1.jsonl \
--lr 1e-5 --epochs 10 \
--eval_epoch 1.0 --report_epoch 1.0 \
--exp_dir <base_dir>/exp/step_scorer/$model \
--code_dir $code_dir \
--save_model 

