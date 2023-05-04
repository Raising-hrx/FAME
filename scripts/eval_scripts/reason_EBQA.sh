cd FAME/code

base_dir="FAME"
code_dir="${base_dir}/code"

# ----- data -----
data_split="test"
data_path="${base_dir}/data/EntailmentBankQA/dataset/task_3/${data_split}.jsonl"
task='QA'

# ----- retriever -----
corpus_path="${base_dir}/data/entailment_trees_emnlp2021_data_v3/supporting_data/preprocessed_corpus.json"
retriever_path_or_name="${base_dir}/exp/retriever/v1"

# ----- entailment module -----
entailment_module_exp_dir="${base_dir}/exp/EntailmentModule/IK3tEKPo/"

# ----- verifier -----
bleurt_path="../bleurt/bleurt-large-512"
step_scorer_exp_dir="${base_dir}/exp/step_scorer/z9TPfknY/"

# ----- important paramaters -----
search_alg="MCTS"
puct=0.2
num_simulations=30

# ----- controller -----
controller_exp_dir="${base_dir}/exp/Controller/Iter5/5Vwv6zll"
controller_model_name="step7614_model.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 python Reason.py --seed 0 \
--task $task --data_split $data_split --data_path $data_path \
--corpus_path $corpus_path --retriever_path_or_name $retriever_path_or_name \
--entailment_module_exp_dir $entailment_module_exp_dir \
--bleurt_path $bleurt_path --step_scorer_exp_dir $step_scorer_exp_dir \
--controller_exp_dir $controller_exp_dir --controller_model_name $controller_model_name \
--search_alg $search_alg --puct $puct --num_simulations $num_simulations \
--max_height 3 \
--code_dir $code_dir \
--save_dir ${controller_exp_dir}/Reproduction/EBQA_test \
--num_process 4 --process_pre_gpu 1 --multiprocess_gpu_ids 0 1 2 3