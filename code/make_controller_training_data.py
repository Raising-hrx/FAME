import os
import sys
import random
import copy
import os.path as osp
import json
import argparse
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

import numpy as np
import transformers
from transformers import T5Tokenizer


from tree_utils import *

from Retriever import Dense_Retriever
from sentence_transformers import SentenceTransformer, CrossEncoder

from RL_agent import oracle_strategy_next_action
from RL_env import EntailmentTreeEnv, State, Action


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='')

    # common argument
    parser.add_argument('--linearize_state_form', type=str, default='QAHPS')
    parser.add_argument('--max_token_length', type=int, default=510)
    parser.add_argument('--data_path', type=str, default='')
    
    # from gold
    parser.add_argument('--run_from_gold', action='store_true', default=False)
    parser.add_argument('--max_state_per_sample', type=int, default=20)
    parser.add_argument('--task_1', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)

    # from imitation
    parser.add_argument('--run_from_imitation', action='store_true', default=False)
    parser.add_argument('--corpus_path', type=str, default="")
    parser.add_argument('--retriever_path_or_name', type=str, default="")
    parser.add_argument('--retrieve_budget', type=int, default=10)
    parser.add_argument('--retrieve_top_n', type=int, default=25)
    parser.add_argument('--max_height', type=int, default=5)
    parser.add_argument('--use_entailmentwriter_S', action='store_true', default=False)

    # for final state
    parser.add_argument('--run_final_state', action='store_true', default=False)
    parser.add_argument('--end_linearize_state_form', type=str, default='QAHPN')

    # for verified state
    parser.add_argument('--run_verified_state', action='store_true', default=False)
    parser.add_argument('--thre', type=float, default=0.95)

    # save
    parser.add_argument('--save_path', type=str, default="")


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args




def collect_states_positive_MetGen(input_tree, show = False):
    """
    From MetGen
    modify: we now do not require the step must be 2-premise
    """
    
    input_tree = input_tree
    input_steps = [[sorted(node['pre']),node['id']] for node in input_tree if node['pre']]
    
    print_node_tree(input_tree[0]['id'],input_tree) if show else None
    print("="*20) if show else None
    
    # add depth_to_finish
    def add_depth_to_finish(idx, node_list, dtf = 0):
        node = get_node(idx,node_list)
        node['depth_to_finish'] = dtf
        for pre_idx in node['pre']:
            add_depth_to_finish(pre_idx, node_list, dtf+1)

    add_depth_to_finish(input_tree[0]['id'],input_tree)

    # collect all intermediate state
    hypot_idx = input_tree[0]['id']
    all_states = []

    for n in range(1,len(input_steps)+1):
        for selected_steps in list(combinations(input_steps, n)):
            tmp_node_ids = set()
            for step_pre, step_con in selected_steps:
                for pre_idx in step_pre:
                    tmp_node_ids.add(pre_idx)
                tmp_node_ids.add(step_con)

            if hypot_idx not in tmp_node_ids:
                continue

            # verify whether the selected steps can make a tree
            tmp_tree = []
            for idx in tmp_node_ids:
                node = copy.deepcopy(get_node(idx,input_tree))
                if node['pre'] == []:
                    tmp_tree.append(node)
                elif all([pre_idx in tmp_node_ids for pre_idx in node['pre']]):
                    tmp_tree.append(node)
                else:
                    node['pre'] = []
                    tmp_tree.append(node)


            tmp_tree = get_tree(hypot_idx,tmp_tree)

            depth_to_finish = max([node['depth_to_finish'] for node in tmp_tree])
            length_to_finish = len([[sorted(node['pre']),node['id']] for node in tmp_tree if node['pre']])

            if tmp_tree not in all_states:
                all_states.append(tmp_tree)


    states_positive = []
    for state_tree in all_states:
        
        print_node_tree(state_tree[0]['id'],state_tree) if show else None
        
        pre = sorted(get_leaves_ids(state_tree[0]['id'],state_tree))
        con = state_tree[0]['id'] 
        depth_to_finish = max([node['depth_to_finish'] for node in state_tree])
        
        # future steps
        state_future_steps = [[sorted(node['pre']),node['id']] for node in state_tree if node['pre']]
        length_to_finish = len(state_future_steps)
        
        # previous steps
        state_previous_steps = []
        for step in input_steps:
            if step not in state_future_steps:
                state_previous_steps.append(step)

        gold_next_steps = []
        for step_pre, step_con in state_future_steps:
            if all([idx in pre for idx in step_pre]):
                if len(step_pre) == 1:
                    continue
                else:
                    gold_next_steps.append(step_pre)
                            
        gold_next_steps = [sorted(step) for step in gold_next_steps]
        length_to_finish = len(pre)-1 
        
        gold_next_steps_abd = []
        step_to_con = [future_step[0] for future_step in state_future_steps if future_step[1]==con][0]
        for p_ in pre:
            if p_ in step_to_con:
                gold_next_steps_abd.append([con, p_])
            
        
        
        states_positive.append({
            'pre':sorted(pre),
            'con':con,
            'depth_to_finish':depth_to_finish,
            'length_to_finish':length_to_finish,
            'state_future_steps':state_future_steps,
            'state_previous_steps':state_previous_steps,
            'state_label': 1,
            'fact_label': {idx:1 for idx in pre},
            'fact_dtf':{node['id']:node['depth_to_finish'] for node in state_tree}, # fact depth to finish
            'gold_next_steps':[sorted(step) for step in gold_next_steps],
            'gold_next_steps_abd':gold_next_steps_abd,
        })

    states_positive.append({
        'pre':[hypot_idx],
        'con':hypot_idx,
        'depth_to_finish':0,
        'length_to_finish':0,
        'state_future_steps':[],
        'state_previous_steps':input_steps,
        'state_label': 1,
        'fact_label': {hypot_idx:1},
        'fact_dtf':{hypot_idx:0}, # fact depth to finish
        'gold_next_steps':[],
        'false_next_steps':[],
        'gold_next_steps_abd':[],
    })
    
    return states_positive





def get_from_gold_training_datas(args):
    """
    we shred the gold tree to several intermediate reasoning state
    we do not consider retrieval here
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-large', local_files_only=True)

    linearize_state_form = args.linearize_state_form
    max_token_length = args.max_token_length
    max_state_per_sample = args.max_state_per_sample
    random.seed(args.seed)

    datas_task2 = load_entailmentbank('task_2', 'train')

    training_datas = []

    for data_item in tqdm(datas_task2):

        id2sent = get_id2sent(data_item)
        sent2id = {sent:sent_id for sent_id, sent in id2sent.items()}
        
        # collect all intermediate states
        gold_tree = get_gt_node_list(data_item)
        metgen_positive_states = collect_states_positive_MetGen(gold_tree, show = False)
        
        # We limit the number of states produced by each sample
        if len(metgen_positive_states) > max_state_per_sample:
            random.shuffle(metgen_positive_states)
            metgen_positive_states = metgen_positive_states[:max_state_per_sample]
        
        
        for metgen_state in metgen_positive_states:
            
            # -------- convert the metgen_state to State --------
            S = [id2sent[p_id] for p_id in metgen_state['pre']]

            if args.task_1:
                pass
            else:
                distractors = [id2sent[sent_id] for sent_id in data_item['meta']['distractors']]
                S += distractors

            P = []
            used_S = []
            for step in metgen_state['state_previous_steps'][::-1]:
                P.append({
                    'pre_sent': [id2sent[p_id] for p_id in step[0]],
                    'con_sent': id2sent[step[1]]
                })
                used_S +=  [id2sent[p_id] for p_id in step[0]]
                
            while True:
                state = State({
                    'H': data_item['hypothesis'],
                    'Q': data_item['question'],
                    'A': data_item['answer'],
                    'S': S,
                    'P': P,
                    'used_S': used_S,
                    'sent2id': sent2id,
                })
                controller_src = state.linearize_state(form = linearize_state_form)
                if len(tokenizer(controller_src)['input_ids']) < max_token_length:
                    break
                else:
                    S = S[:-1]

            state.reassign_sent2id()
            
            # -------- gold actions --------
            gold_actions = []
            if 'hypothesis' in metgen_state['pre']:
                gold_actions.append({'type': Action.end, 'is_proved': 'proved'})
            else:
                for step in metgen_state['state_future_steps']:
                    pre_sent = [id2sent[p_id] for p_id in step[0]]
                    
                    # check if all premises in S
                    if not all([p_ in state.S for p_ in pre_sent]):
                        continue
                    
                    gold_actions.append({
                        'type': Action.reason,
                        'step': {
                            'pre_sent': pre_sent,
                            'con_sent': id2sent[step[1]],
                        }
                    })
                    
                for ac in gold_actions:
                    ac['step']['pre_id'] = [state.sent2id[sent] for sent in ac['step']['pre_sent']]
            
            # -------- make training item --------
            for ac in gold_actions:
                if ac['type'] == Action.end:
                    continue
                    
                for p_ in ac['step']['pre_sent']:
                    assert p_ in state.S, print(ac, p_, state.S)
                    
                training_item = {
                    'src': state.linearize_state(form = linearize_state_form),
                    'tgt': Action.linearize_action(ac),
                    'state': state.to_dict(),
                    'tgt_action': ac,
                }
                training_datas.append(training_item)

    return training_datas


def get_from_imitation_training_datas(args):
    tokenizer = T5Tokenizer.from_pretrained('t5-large', local_files_only=True)

    linearize_state_form = args.linearize_state_form
    use_entailmentwriter_S = args.use_entailmentwriter_S
    action_budget = {
        Action.retrieve: args.retrieve_budget,
    }
    retrieve_top_n = args.retrieve_top_n
    max_token_length = args.max_token_length

    max_height = args.max_height


    # Retriever
    print(f"Loading corpus from {args.corpus_path}")
    corpus = json.load(open(args.corpus_path))

    print(f"Loading Retriever from {args.retriever_path_or_name} \n")
    bi_encoder = SentenceTransformer(args.retriever_path_or_name)
    retriever = Dense_Retriever(corpus, bi_encoder, buffer_file = None, device='cuda')

    datas = load_entailmentbank('task_1', 'train')
    datas_task3 = load_entailmentbank('task_3', 'train')


    entailment_env = EntailmentTreeEnv(retriever = retriever, 
                                        entailment_module = None, 
                                        verifier = None,
                                        env_args = {
                                            'action_budget': action_budget,
                                            'retrieve_top_n': retrieve_top_n,
                                            'max_height': max_height,
                                        })

    training_datas = []

    for item_i, data_item in enumerate(datas):
        print(item_i)
        
        entailment_env.reset(data_item)
        entailment_env.state.update_S([])
        
        if use_entailmentwriter_S:
            data_item_task3 = datas_task3[item_i]
            entailmentwriter_S = list(data_item_task3['meta']['triples'].values())
            entailment_env.state.update_S(entailmentwriter_S)
        else:
            entailment_env.step(Action.parse_action("retrieve: hypothesis"))
            
            
        while True:
            
            # check token length
            while True:
                controller_src = entailment_env.state.linearize_state(form = linearize_state_form)
                if len(tokenizer(controller_src)['input_ids']) < max_token_length:
                    break
                else:
                    entailment_env.state.S = entailment_env.state.S[:-1]
            
            entailment_env.state.reassign_sent2id()

            # -------- gold actions --------
            oracle_action = oracle_strategy_next_action(entailment_env.state, retriever, 
                                                        retrieve_top_n = retrieve_top_n)

            # for Action.retrieve, filter by priority
            if oracle_action['type'] == Action.retrieve and oracle_action['query_priority'] < 0.3:
                break   

            if oracle_action['type'] == Action.end:
                break

            x = 0
            for ac in entailment_env.state.history_actions[::-1]:
                if ac['type'] == Action.retrieve:
                    x += 1
                else:
                    break
            if x > 3:
                print(f"continuously retrieve for {x} times")
                break
    
    
            # -------- make training item --------
            training_item = {
                'src': entailment_env.state.linearize_state(form = linearize_state_form),
                'tgt': Action.linearize_action(oracle_action),
                'state': copy.deepcopy(entailment_env.state.to_dict()),
                'tgt_action': copy.deepcopy(oracle_action),
            }
            training_datas.append(training_item)
            
            state,_, done,_ = entailment_env.step(oracle_action)
            if done:
                break

    return training_datas


def get_final_state_training_datas(datas, end_linearize_state_form):

    datas_task1 = load_entailmentbank('task_1', 'train')
    id2gold_item = {item['id']:item for item in datas_task1}

    training_datas = []
    for data_item in datas:
        for choice in data_item['choices']:
            pred_state = State(choice['pred_state']) # the final state
            pred_state.choices_str = State(data_item).choices_str

            # error choice
            if choice['correct_answer'] == False:
                action = {'type': Action.end, 'is_proved': 'unproved'}
                pred_state.reassign_sent2id()
                training_item = {
                    'src': pred_state.linearize_state(end_linearize_state_form),
                    'tgt': Action.linearize_action(action),
                    'state': copy.deepcopy(choice['pred_state']),
                    'tgt_action': copy.deepcopy(action),
                    'choice': copy.deepcopy(choice), 
                }
                training_datas.append(training_item)
                
            # correct choice
            else:
                action = {'type': Action.end, 'is_proved': 'proved'}
                    
                # ----- add action end -----
                pred_state.reassign_sent2id()
                training_item = {
                    'src': pred_state.linearize_state(end_linearize_state_form),
                    'tgt': Action.linearize_action(action),
                    'state': copy.deepcopy(choice['pred_state']),
                    'tgt_action': copy.deepcopy(action),
                    'choice': copy.deepcopy(choice), 
                }
                training_datas.append(training_item)
    
                    
                    
                # ----- replace the P with correct proof -----
                # state = State(data_item)
                state = State(id2gold_item[data_item['id']])
                state.P = []
                gold_tree = get_gt_node_list(id2gold_item[data_item['id']])
                gold_id2sent = get_id2sent(id2gold_item[data_item['id']])
                
                for node in gold_tree[::-1]:
                    if node['id'].startswith('sent'):
                        state.sent2id[node['sent']] = node['id']
                    elif node['id'].startswith('int'):
                        state.sent2id[node['sent']] = node['id']
                        state.P.append({
                            'pre_sent': [gold_id2sent[p] for p in node['pre']],
                            'con_sent': node['sent'],
                        })
                    elif node['id'] == 'hypothesis':
                        state.sent2id[node['sent']] = state.next_id('int')
                        state.P.append({
                            'pre_sent': [gold_id2sent[p] for p in node['pre']],
                            'con_sent': node['sent'],
                        })

                state.H = choice['hypothesis'] # use the QA2D hypothesis
                state.sent2id[choice['hypothesis']] = 'hypothesis'
                
                state.reassign_sent2id()
                training_item = {
                    'src': state.linearize_state(end_linearize_state_form),
                    'tgt': Action.linearize_action(action),
                    'state': copy.deepcopy(state.to_dict()),
                    'tgt_action': copy.deepcopy(action),
                    'choice': copy.deepcopy(choice), 
                }

                training_datas.append(training_item)

    return training_datas

def get_verified_state_training_datas(datas, linearize_state_form, thre):

    training_datas = []
    for data_item in datas:
        for choice in data_item['choices']:

            # correct choice
            if choice['correct_answer'] == True:

                # if the final state is valid
                if choice['pred_state_verifier_score'] > thre:
                    pred_trace = choice['pred_trace'] # list of (state, action)
                    
                    for state_, action in pred_trace:
                        state = State(state_)
                        
                        if len(state.S) == 0:
                            continue
                        if action['type'] == Action.end:
                            continue
                    
                        training_item = {
                            'src': state.linearize_state(linearize_state_form),
                            'tgt': Action.linearize_action(action),
                            'state': copy.deepcopy(state.to_dict()),
                            'tgt_action': copy.deepcopy(action),
                            'choice': copy.deepcopy(choice), 
                        }

                        training_datas.append(training_item)

    return training_datas

if __name__ == '__main__':

    args = get_params()
    
    if args.run_from_gold:
        training_datas = get_from_gold_training_datas(args)

        print(f"Saving data to {args.save_path}")
        with open(args.save_path, 'w') as f:
            for i in training_datas:
                f.writelines(json.dumps(i)+'\n')
    
    elif args.run_from_imitation:
        training_datas = get_from_imitation_training_datas(args)

        print(f"Saving data to {args.save_path}")
        with open(args.save_path, 'w') as f:
            for i in training_datas:
                f.writelines(json.dumps(i)+'\n')
    
    elif args.run_final_state:
        print(f"Loading data from {args.data_path}")
        datas = [json.loads(line) for line in open(args.data_path).readlines()]
        training_datas = get_final_state_training_datas(datas, args.end_linearize_state_form)
        
        print(f"Saving data to {args.save_path}")
        with open(args.save_path, 'w') as f:
            for i in training_datas:
                f.writelines(json.dumps(i)+'\n')

    elif args.run_verified_state:
        print(f"Loading data from {args.data_path}")
        datas = [json.loads(line) for line in open(args.data_path).readlines()]
        training_datas = get_verified_state_training_datas(datas, args.linearize_state_form, args.thre)
        
        print(f"Saving data to {args.save_path}")
        with open(args.save_path, 'w') as f:
            for i in training_datas:
                f.writelines(json.dumps(i)+'\n')

    else:
        raise NotImplementedError
