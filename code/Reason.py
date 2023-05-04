import json
import os
import argparse
import random
import sys
import os.path as osp
from multiprocessing import Pool
import copy
import math

# import transformers
# import numpy as np
# import torch

# from tqdm import tqdm
# from collections import defaultdict, Counter

from tree_utils import *
# from Retriever import BM25_Retriever, Dense_Retriever
# from sentence_transformers import SentenceTransformer, CrossEncoder

# from RL_agent import oracle_strategy_next_action, interact
# from RL_env import EntailmentTreeEnv, State, Action

# from evaluate_metric import eval_tree_task3, collect_results
# from Controller import Controller

# from Verifier import Verifier
# from StepScorer import StepScorer
# from EntailmentModule import EntailmentModule

# import bleurt 
# from bleurt import score
# import tensorflow as tf
# for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)
    


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    # dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training BART')

    # dateset
    parser.add_argument("--task", type=str, default='QA') 
    parser.add_argument("--data_path", type=str) 
    parser.add_argument("--initial_S", type=str, default=None) 
    parser.add_argument("--data_split", type=str, default='test') 

    
    # model
    parser.add_argument("--bleurt_path", type=str, default="bleurt/bleurt-large-512")  
    
    parser.add_argument("--corpus_path", type=str, default="")  
    parser.add_argument("--retriever_path_or_name", type=str, default="")  
    parser.add_argument("--retriever_buffer_path", type=str, default=None)  

    parser.add_argument("--controller_exp_dir", type=str, default="")  
    parser.add_argument("--controller_model_name", type=str, default="")  
    parser.add_argument("--linearize_state_form", type=str, default="QAHPS") 
    parser.add_argument("--end_linearize_state_form", type=str, default="QAHPN") 
    parser.add_argument("--controller_num_return_sequences", type=int, default=5) 

    # verifier
    parser.add_argument("--P_score_type", type=str, default='mean')  
    parser.add_argument("--H_score_type", type=str, default='bleurt+step_scorer')  
    parser.add_argument("--merge_strategy", type=str, default='P+H')  


    parser.add_argument("--step_scorer_exp_dir", type=str, default="")  

    parser.add_argument("--entailment_module_exp_dir", type=str, default="")  
    parser.add_argument("--entailment_module_buffer_path", type=str, default=None)  

    # environment
    parser.add_argument("--retrieve_top_n", type=int, default=25) 
    parser.add_argument("--retrieve_budget", type=int, default=10) 
    parser.add_argument("--max_height", type=int, default=5) 
    parser.add_argument("--check_premise_overlap", type=int, default=0) # 1 for True
    parser.add_argument("--step_score_thre", type=float, default=0.0) 
    parser.add_argument("--module_num_return", type=int, default=5) 


    # reasoning strategy
    parser.add_argument("--search_alg", type=str, default="MCTS")  
    parser.add_argument("--force_reason_strategy", type=str, default="after_retrieve") 
    parser.add_argument("--choice_select_strategy", type=str, default="V+C")  

    # search_alg: A* / lookahead /None
    parser.add_argument("--lookahead_L", type=int, default=0)  
    parser.add_argument("--action_score_strategy", type=str, default=None)  
    # search_alg: MCTS
    parser.add_argument("--puct", type=float, default=0.2) 
    parser.add_argument("--num_simulations", type=int, default=30) 
    # search_alg: beam
    parser.add_argument("--beam_size", type=int, default=1) 
    # parser.add_argument("--num_simulations", type=int, default=30) 

    # save
    parser.add_argument("--code_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=3407)

    # multiprocessing
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--process_pre_gpu", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--multiprocess_gpu_ids", type=int, nargs='*', default=[])
    parser.add_argument("--min_r", type=int, default=None)
    parser.add_argument("--max_r", type=int, default=None)



    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args


def run(args):

    # set gpu
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # import all these after setting the 'CUDA_VISIBLE_DEVICES'
    import transformers
    import numpy as np
    import torch

    from tqdm import tqdm
    from collections import defaultdict, Counter

    from Retriever import BM25_Retriever, Dense_Retriever
    from sentence_transformers import SentenceTransformer, CrossEncoder

    from RL_agent import oracle_strategy_next_action, interact, MCTS_Searcher, Beam_Searcher
    from RL_env import EntailmentTreeEnv, State, Action

    from evaluate_metric import eval_tree_task3, collect_results
    from Controller import Controller

    from Verifier import Verifier
    from StepScorer import StepScorer
    from EntailmentModule import EntailmentModule

    import bleurt 
    from bleurt import score
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # print("CUDA_VISIBLE_DEVICES", os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.is_available())

    # set random seed 
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    
    # load bleurt
    print(f"Loading BLEURT model from {args.bleurt_path}")
    bleurt_scorer = score.BleurtScorer(args.bleurt_path) 

    # Retriever
    print(f"Loading corpus from {args.corpus_path}")
    corpus = json.load(open(args.corpus_path))

    print(f"Loading Retriever from {args.retriever_path_or_name} \n buffer {args.retriever_buffer_path}")
    bi_encoder = SentenceTransformer(args.retriever_path_or_name)
    buffer_file = args.retriever_buffer_path
    retriever = Dense_Retriever(corpus, bi_encoder, buffer_file = buffer_file, device='cuda')

    # Controller
    print(f"Loading Controller from {args.controller_exp_dir}  {args.controller_model_name}")
    controller = Controller(args.controller_exp_dir, args.controller_model_name, device = 'cuda')


    # Verifier
    print(f"Loading StepScorer from {args.step_scorer_exp_dir}")
    step_scorer = StepScorer(args.step_scorer_exp_dir, device='cuda')

    # tmp: entail_scorer
    entail_scorer = None

    # verifier = Verifier(step_scorer=step_scorer, bleurt_scorer=bleurt_scorer)
    verifier = Verifier(step_scorer=step_scorer, 
                        entail_scorer = entail_scorer, 
                        bleurt_scorer=bleurt_scorer, 
                        P_score_type = args.P_score_type, 
                        H_score_type = args.H_score_type, 
                        merge_strategy = args.merge_strategy)
    # Entailment Module
    print(f"Loading EntailmentModule from {args.entailment_module_exp_dir} \n buffer {args.entailment_module_buffer_path}")
    entailment_module = EntailmentModule(exp_dir = args.entailment_module_exp_dir, 
                                        buffer_file = args.entailment_module_buffer_path, 
                                        model_name='best_model.pth', device = 'cuda')


    action_budget = {
        Action.retrieve: args.retrieve_budget,
    }

    # controller generate_args
    generate_args = {
        'num_beams': args.controller_num_return_sequences,
        'num_return_sequences': args.controller_num_return_sequences,
    }
    constraints_generate_args = {
        'num_beams': args.controller_num_return_sequences,
        'num_return_sequences': args.controller_num_return_sequences,
        'constraints': [transformers.PhrasalConstraint(controller.tokenizer("reason", add_special_tokens=False).input_ids)],
    }

    entailment_env = EntailmentTreeEnv(retriever = retriever, 
                                    entailment_module = entailment_module, 
                                    verifier = verifier,
                                    env_args = {
                                        'action_budget':action_budget,
                                        'retrieve_top_n':args.retrieve_top_n,
                                        'max_height':args.max_height,
                                        'check_premise_overlap':args.check_premise_overlap,
                                        'step_score_thre': args.step_score_thre,
                                        'module_num_return': args.module_num_return,
                                    }
                                    )

    interact_args = {
        'linearize_state_form': args.linearize_state_form,
        'controller_generate_args': generate_args,
        'constraints_args_reason': constraints_generate_args,
        'force_reason_strategy': args.force_reason_strategy,
        'action_score_strategy': args.action_score_strategy,
        'lookahead_L': args.lookahead_L,
        'puct': args.puct,
        'beam_size': args.beam_size,
    }


    # Load data
    print(f"Loading data from {args.data_path}")
    datas = [json.loads(line) for line in open(args.data_path).readlines()]
    if (args.min_r is not None) and (args.max_r is not None):
        min_idx = math.ceil(len(datas)*args.min_r)
        max_idx = math.ceil(len(datas)*args.max_r)
        datas = datas[min_idx:max_idx]

    if args.task in ['QA', 'WTQA', 'OBQA']:
        print(f"GPU: {args.gpu_id}")

        # reason with controller
        for data_item in tqdm(datas):
            for choice_idx in range(len(data_item['choices'])):

                choice = data_item['choices'][choice_idx]
                print(f"{args.task} {data_item['id']} {choice['text']}")

                choice_state = {
                    'Q': data_item['question'],
                    'A': choice['text'],
                    'H': choice['hypothesis'],
                    'choices': data_item['choices'],
                }
                
                entailment_env.reset(choice_state)

                if args.initial_S in ['None', None]:
                    entailment_env.step(Action.parse_action("retrieve: hypothesis"))
                elif args.initial_S in ['use_item']:
                    item_S = list(data_item['meta']['triples'].values())
                    entailment_env.state.update_S(item_S)
                    print(f"Using item initial S: {len(item_S)}")
                elif args.initial_S in ['OBQA_use_item_only']:
                    item_S = list(choice['entailer_facts'])
                    entailment_env.state.update_S(item_S)
                    print(f"Using choice entailer_facts: {len(item_S)}")
                elif args.initial_S in ['OBQA_use_item']:
                    entailment_env.step(Action.parse_action("retrieve: hypothesis"))
                    item_S = list(choice['entailer_facts'])
                    new_S = item_S + entailment_env.state.S[:10]
                    entailment_env.state.update_S(new_S)
                    print(f"Using choice entailer_facts and retrieve: {len(item_S)} -> {len(new_S)} ")
                else:
                    raise NotImplemented

                # print("*"*50)
                # print(data_item['meta']['triples'])
                # print(entailment_env.state)

                if args.search_alg in ['None', 'A']:
                    entailment_env, extra_info = interact(entailment_env, controller, interact_args)
                elif args.search_alg in ['MCTS']:
                    searcher = MCTS_Searcher(interact_args)
                    # print(f"args.num_simulations: {args.num_simulations}")
                    entailment_env, extra_info = searcher.run(entailment_env, controller, num_simulations = args.num_simulations)
                elif args.search_alg in ['beam']:
                    searcher = Beam_Searcher(interact_args)
                    # print(f"args.num_simulations: {args.num_simulations}")
                    entailment_env, extra_info = searcher.run(entailment_env, controller, num_simulations = args.num_simulations)
                else:
                    raise NotImplemented

                data_item['choices'][choice_idx]['pred_state'] = entailment_env.state.to_dict()
                veifier_info = entailment_env.verifier.verify_a_state(entailment_env.state, return_dict = True)
                data_item['choices'][choice_idx]['pred_state_verifier_info'] = veifier_info
                data_item['choices'][choice_idx]['pred_state_verifier_score'] = veifier_info['state_score']

                # print(entailment_env.state.linearize_state(args.end_linearize_state_form))
                data_item['choices'][choice_idx]['controller_proved_score'] = controller.compute_proved_score([entailment_env.state.linearize_state(args.end_linearize_state_form)])[0]

                try:
                    root_node = verifier.verify_get_root_node(entailment_env.state)
                    pred_tree = entailment_env.state.get_pred_node_list(root_node = root_node)
                except:
                    print("FAILED: pred_tree = state.get_pred_node_list()")
                    pred_tree = [
                        {'id':'hypothesis', 'sent':data_item['hypothesis'], 'pre':[]},
                    ]
                data_item['choices'][choice_idx]['pred_tree'] = pred_tree

                data_item['choices'][choice_idx]['pred_trace'] = entailment_env.history_trace

                data_item['choices'][choice_idx]['extra_info'] = {'excuted_action_num': extra_info['excuted_action_num']}
                

            # select choice
            if args.choice_select_strategy in ['None', None, 'verifier', 'V']:
                sorted_choices = sorted(data_item['choices'],key=lambda x:x['pred_state_verifier_score'],reverse=True)
            
            elif args.choice_select_strategy in ['controller', 'C']:
                sorted_choices = sorted(data_item['choices'],key=lambda x:x['controller_proved_score'],reverse=True)

            elif args.choice_select_strategy in ['V+C', 'C+V']:
                sorted_choices = sorted(data_item['choices'],key=lambda x:(x['pred_state_verifier_score']+x['controller_proved_score']),reverse=True)

            else:
                raise NotImplemented

            selected_choice = sorted_choices[0]
            data_item['selected_choice'] = selected_choice


    elif args.task == 'EB':

        for data_item in tqdm(datas):

            print(f"{args.task} {data_item['id']}")

            entailment_env.reset(data_item)
            entailment_env.state.update_S([])
            entailment_env.step(Action.parse_action("retrieve: hypothesis"))
            
            # entailmentwriter_S = list(data_item['meta']['triples'].values())
            # entailment_env.state.update_S(entailmentwriter_S)

            if args.search_alg in ['None', 'A']:
                entailment_env, extra_info = interact(entailment_env, controller, interact_args)
            elif args.search_alg in ['MCTS']:
                searcher = MCTS_Searcher(interact_args)
                # print(f"args.num_simulations: {args.num_simulations}")
                entailment_env, extra_info = searcher.run(entailment_env, controller, num_simulations = args.num_simulations)
            else:
                raise NotImplemented

            data_item['pred_state'] = entailment_env.state.to_dict()

            try:
                root_node = verifier.verify_get_root_node(entailment_env.state)
                pred_tree = entailment_env.state.get_pred_node_list(root_node = root_node)
            except:
                print("FAILED: pred_tree = state.get_pred_node_list()")
                pred_tree = [
                    {'id':'hypothesis', 'sent':data_item['hypothesis'], 'pre':[]},
                ]
            data_item['pred_tree'] = pred_tree

    else:
        raise NotImplemented

    return datas

def eval_func(datas, args):
    from RL_env import State

    # load bleurt
    import bleurt 
    from bleurt import score
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Loading BLEURT model from {args.bleurt_path}")
    bleurt_scorer = score.BleurtScorer(args.bleurt_path) 

    import numpy as np
    from collections import defaultdict


    eval_result = {}
    if args.task == 'QA':

        qa_metrics = {}

        answer_correctness_by_id = {}
        answer_correctness_by_diff = defaultdict(list)
        all_pred_info_for_selected_answer = []

        for data_item in datas:
            # select choices
            selected_choice = data_item['selected_choice']

            pred_state = State(selected_choice['pred_state'])
            pred_tree = selected_choice['pred_tree']
            
            # collect answer correctness
            answer_correctness_by_id[data_item['id']] = selected_choice['correct_answer']
            difficulty = data_item.get('arc_item', {}).get('difficulty', 'unknown')
            answer_correctness_by_diff[difficulty].append(selected_choice['correct_answer'])
            
            # collect proof
            if selected_choice['correct_answer'] == True:
                pred_tree[0]['sent'] = data_item['hypothesis']
                pred_tree[0]['id'] = 'hypothesis'
                id2sent = {node['id']:node['sent'] for node in pred_tree}
                pred_info = {
                    'proof': linearize_node_list(pred_tree),
                    'id2sent': id2sent,
                    'id': data_item['id'],
                    'hypothesis': data_item['hypothesis'],
                }
                all_pred_info_for_selected_answer.append(pred_info)
                
            else:
                pred_info = {
                    'proof': linearize_node_list(pred_state.get_pred_node_list()),
                    'id2sent': {v:k for k,v in pred_state.sent2id.items()},
                    'id': data_item['id'],
                    'hypothesis': data_item['hypothesis'],
                }
                all_pred_info_for_selected_answer.append(pred_info)

        # compute metrics
        qa_metrics['answer_correctness_by_id'] = answer_correctness_by_id
        qa_metrics['all_pred_info_for_selected_answer'] = all_pred_info_for_selected_answer

        qa_metrics['answer_acc'] = np.mean(list(answer_correctness_by_id.values()))
        qa_metrics['answer_acc_by_diff'] = {k:np.mean(v) for k,v in answer_correctness_by_diff.items()}

        eval_result = qa_metrics


    elif args.task == 'EB':
        all_pred_info = []
        for item_i, data_item in enumerate(datas):
            pred_state = State(data_item['pred_state'])
            pred_tree = data_item['pred_tree']
            
            all_pred_info.append({
                'proof': linearize_node_list(pred_tree),
                'id2sent': {v:k for k,v in pred_state.sent2id.items()},
                'id': data_item['id'],
                'hypothesis': data_item['hypothesis'],
            })
        # collected = offical_evaluation_api('task_3', args.data_split, all_pred_info, bleurt_scorer)
        collected = all_pred_info
        eval_result = collected
    
    elif args.task == 'WTQA':
        qa_metrics = {}

        answer_correctness_by_id = {}
        answer_correctness_by_diff = defaultdict(list)

        for data_item in datas:
            # select choices
            selected_choice = data_item['selected_choice']

            pred_state = State(selected_choice['pred_state'])
            pred_tree = selected_choice['pred_tree']
            
            # collect answer correctness
            answer_correctness_by_id[data_item['id']] = selected_choice['correct_answer']
            difficulty = data_item.get('difficulty', 'unknown')
            answer_correctness_by_diff[difficulty].append(selected_choice['correct_answer'])
        
        # compute metrics
        qa_metrics['answer_correctness_by_id'] = answer_correctness_by_id
        qa_metrics['answer_acc'] = np.mean(list(answer_correctness_by_id.values()))
        qa_metrics['answer_acc_by_diff'] = {k:np.mean(v) for k,v in answer_correctness_by_diff.items()}

        eval_result = qa_metrics

    elif args.task == 'OBQA':
        qa_metrics = {}

        answer_correctness_by_id = {}

        for data_item in datas:
            # select choices
            selected_choice = data_item['selected_choice']
            
            # collect answer correctness
            answer_correctness_by_id[data_item['id']] = selected_choice['correct_answer']
            difficulty = data_item.get('difficulty', 'unknown')
        
        # compute metrics
        qa_metrics['answer_correctness_by_id'] = answer_correctness_by_id
        qa_metrics['answer_acc'] = np.mean(list(answer_correctness_by_id.values()))

        eval_result = qa_metrics

    else:
        raise NotImplemented   

    return eval_result

if __name__ == '__main__':
    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1,1e4)

    if args.save_dir is None:
        args.save_dir = osp.join(args.controller_exp_dir, f'{args.task}_prediction')
    args.save_dir = osp.join(args.save_dir, get_random_dir_name())
    args.save_dir = f"{args.save_dir}_{args.controller_model_name.split('.')[0]}"

    os.makedirs(args.save_dir, exist_ok=True)

    # backup scripts
    os.system(f'cp -r {args.code_dir} {args.save_dir}')

    # dump config.json
    with open(osp.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.num_process <= 1:
        datas = run(args)
    else:
        # multiprocessing
        print("use multiprocessing")
        assert args.num_process <= len(args.multiprocess_gpu_ids)*args.process_pre_gpu
        args_list = []
        for i in range(args.num_process):
            new_args = copy.deepcopy(args)
            # new_args.gpu_id = math.floor(i/args.process_pre_gpu) + args.start_gpu_id
            new_args.gpu_id = args.multiprocess_gpu_ids[math.floor(i/args.process_pre_gpu)]
            new_args.min_r = i * (1.0 / args.num_process)
            new_args.max_r = (i + 1) * (1.0 / args.num_process)
            args_list.append(new_args)

        with Pool(processes=args.num_process) as pool:
            results = pool.map(run, args_list)

        datas = []
        for r in results:
            datas += r 

    # save result
    print(f"Saving results")
    with open(osp.join(args.save_dir, 'result.jsonl'), 'w') as f:
        f.writelines([json.dumps(data_item)+'\n' for data_item in datas])

    if args.data_split in ['dev', 'test']:
        # eval
        eval_result = eval_func(datas, args)
        with open(osp.join(args.save_dir, 'eval_result.jsonl'), 'w') as f:
            json.dump(eval_result, f)
    
    elif args.data_split in ['train']:
        from make_controller_training_data import get_final_state_training_datas, get_verified_state_training_datas
        
        training_datas = get_final_state_training_datas(datas, args.end_linearize_state_form)
        with open(osp.join(args.save_dir, 'final_state_training_datas.jsonl'), 'w') as f:
            f.writelines([json.dumps(data_item)+'\n' for data_item in training_datas])

        for thre in [0.98, 0.95, 0.9]:
            training_datas = get_verified_state_training_datas(datas, args.linearize_state_form, thre)
            with open(osp.join(args.save_dir, f'verified_state_training_datas.{thre}.jsonl'), 'w') as f:
                f.writelines([json.dumps(data_item)+'\n' for data_item in training_datas])

    else:
        raise NotImplemented
