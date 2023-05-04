cd <base_dir>/code

code_dir=<base_dir>/code
save_dir=<base_dir>/exp/retriever/mpnet

model_name=sentence-transformers/all-mpnet-base-v2


bs=40
lr=1e-5
triple=True
scale=50.0
steps=20000
save_steps=50
query_candidate_type=H
random_neg_pos_ratio=10
hard_neg_pos_ratio=10
seed=7433

gpu=0
exp_dir=<base_dir>/exp/retriever/mpnet_best
CUDA_VISIBLE_DEVICES=$gpu python train_Retriever.py --seed $seed \
--model_name_or_path $model_name \
--scale $scale --triple $triple --query_candidate_type $query_candidate_type \
--bs $bs --lr $lr --steps $steps  --save_steps $save_steps \
--random_neg_pos_ratio $random_neg_pos_ratio --hard_neg_pos_ratio $hard_neg_pos_ratio \
--max_length 256 --num_warmup_steps 100 \
--code_dir $code_dir \
--exp_dir $exp_dir 

