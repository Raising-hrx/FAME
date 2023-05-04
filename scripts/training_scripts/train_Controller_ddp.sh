base_dir=<>
cd ${base_dir}/code

data_dir=${base_dir}/data/ControllerData
code_dir=${base_dir}/code
model=t5-large

### ----- Iteration 0 -----
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 Controller.py \
--model_name_or_path $model --bs 20 \
--train_data $data_dir/controller.train.from_gold.v9.jsonl $data_dir/controller.train.from_imitation.v9.jsonl \
--lr 3e-5 --epochs 30 --adafactor \
--max_src_length 512 --max_tgt_length 32 \
--eval_epoch 5 --report_epoch 0.3 \
--exp_dir ${base_dir}/exp/Controller/Iter0 \
--code_dir $code_dir \
--save_model


### ----- Iteration X -----
### for X > 0, use reason_EBQA.sh to reason on the EntailmentBankQA training split, 
### and then add the filtered prediction results from previous iterations in the train_data

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.launch --nproc_per_node=4 Controller.py \
# --model_name_or_path $model --bs 20 \
# --resume_path ${base_dir}/exp/Controller/Iter4/***/***.pth \
# --train_data $data_dir/controller.train.from_gold.v9.jsonl $data_dir/controller.train.from_imitation.v9.jsonl \
# ******
# ${base_dir}/exp/Controller/Iter4/***/***/final_state_training_datas.jsonl \
# ${base_dir}/exp/Controller/Iter4/***/***/verified_state_training_datas.0.98.jsonl \
# --lr 1e-5 --epochs 10 --adafactor \
# --max_src_length 512 --max_tgt_length 32 \
# --eval_epoch 1 --report_epoch 0.3 \
# --exp_dir ${base_dir}/exp/Controller/Iter5 \
# --code_dir $code_dir \
# --save_model
