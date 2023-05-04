from tree_utils import *
import itertools
import copy
import numpy as np
import networkx as nx

import spacy
spacy_nlp = spacy.load("en_core_web_sm")


from evaluate_metric import eval_tree_task3

class Action:
    retrieve = 'retrieve'
    reason = 'reason'
    end = 'end'

    @classmethod
    def linearize_action(cls, action):
        action_str = ""

        if action['type'] == Action.retrieve:
            action_str += f"{action['type']}: {action['query_id']}"

        elif action['type'] == Action.reason:
            action_str += f"{action['type']}: "
            action_str += " & ".join(sorted(action['step']['pre_id']))
            if 'con_sent' in action['step']:
                action_str += f" -> {action['step']['con_sent']}"

        elif action['type'] == Action.end:
            action_str += f"{action['type']}"
            if 'is_proved' in action:
                action_str += f": {action['is_proved']}"

        else:
            raise NotImplementedError

        return action_str

    @classmethod
    def parse_action(cls, action_str):
    
        if ':' in action_str:
            action_type, paras_str = action_str.split(':', maxsplit=1)
            action_type = action_type.strip()
        else:
            action_type = action_str.strip()
            paras_str = None

            
        if action_type == Action.retrieve:
            action = {
                'type': action_type,
                'query_id': paras_str.strip(),
            }
        
        elif action_type == Action.reason:
            if '->' not in paras_str:
                pre_id = [p.strip() for p in paras_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                    },
                    'use_module': True,
                }
            else:
                pre_id_str, con_sent = paras_str.split('->', maxsplit=1)
                pre_id = [p.strip() for p in pre_id_str.split('&')]
                action = {
                    'type': action_type,
                    'step': {
                        'pre_id': pre_id,
                        'con_sent': con_sent.strip(),
                    },
                    'use_module': False,
                }

        elif action_type == Action.end:
            if paras_str is None:
                action = {
                    'type': action_type,
                }
            else:
                action = {
                    'type': action_type,
                    'is_proved': paras_str.strip(),
                }  
        else:
            action = None

        return action

class State:
    def __init__(self, data_item = None):

        self.data_item = data_item

        self.H = data_item.get('H','')
        self.Q = data_item.get('Q','')
        self.A = data_item.get('A','')
        self.S = data_item.get('S', [])
        self.P = data_item.get('P', [])
        self.used_S = data_item.get('used_S', [])
        self.sent2id = data_item.get('sent2id', {})
        self.history_actions = data_item.get('history_actions', [])
        self.choices_str = data_item.get('choices_str', [])

        if not self.H: self.H = data_item['hypothesis']
        if not self.Q: self.Q = data_item['question']
        if not self.A: self.A = data_item['answer']
        if not self.S: self.S = list(data_item.get('meta',{}).get('triples',{}).values())
        if not self.sent2id: self.sent2id = {v:k for k,v in data_item.get('meta',{}).get('triples',{}).items()}
        if not self.choices_str: self.choices_str = " ".join([f"({choice['label']}) {choice['text']}" for choice in data_item.get('choices', [])])

        self.sent2id[self.H] = 'hypothesis'

            
    def expand_P(self, step):
        # move premises from S to used_S
        assert all([p in self.S for p in step['pre_sent']])
        for p in set(step['pre_sent']):
            self.S.remove(p)
            self.used_S.append(p)
        
        # add conclusion to S at index 0
        self.S = [step['con_sent']] + self.S
        
        # add step to P
        self.P.append(step)
        
        # add conclusion to sent2id
        # some intermediate conclusions could be found in the courpus!!!
        if step['con_sent'] != self.H:
            self.sent2id[step['con_sent']] = self.next_id('int')
            

     
    def update_S(self, new_S):
        new_S = [sent for sent in new_S if sent not in self.used_S] # if sent has been used, do not add
        
        remain_S = []
        for sent in set(self.S):
            if not self.sent2id[sent].startswith('sent'):
                remain_S.append(sent)
            else:
                # print(f"del: {self.sent2id[sent]} {sent}")
                del self.sent2id[sent]
        
        # self.S = list(set(remain_S + new_S)) # 'set' will disrupt the order
        self.S = []
        for s in remain_S + new_S:
            if s not in self.S:
                self.S.append(s)

        for sent in self.used_S + self.S:
            if sent not in self.sent2id:
                # print(self.next_id('sent'), self.sent2id.values())
                self.sent2id[sent] = self.next_id('sent')
        
        assert all([sent in self.sent2id for sent in self.S+self.used_S])


    def visualize_gt_graph(self):
        try:
            visualize_nx_graph(get_gt_nx_graph(self.data_item))
        except Exception as e:
            print(e)
            
    def get_pred_nx_graph(self):
        return State.P_to_nx_graph(self.P, self.sent2id)
   
    def get_pred_node_list(self, root_node = None):
        if root_node is None:
            root_node = self.H
        pred_tree = nx_graph_to_node_list(self.get_pred_nx_graph(), root_node = root_node)
        if len(pred_tree) == 0:
            pred_tree = [
                {'id':'hypothesis', 'sent':self.H, 'pre':[]},
            ]
        return pred_tree
    
    def visualize_pred_graph(self):
        try:
            visualize_nx_graph(self.get_pred_nx_graph())
        except Exception as e:
            print(e)

    def get_pred_highest_conclusions(self):
        ### take too much time
        # pred_nx_graph = self.get_pred_nx_graph()
        # return list(set([path[-1] for path in nx.all_topological_sorts(pred_nx_graph) if path]))

        # find nodes which has in_edge but does not have out_edge
        highest_conclusions = []
        nx_graph = self.get_pred_nx_graph()
        for node in nx_graph:
            if nx_graph.in_degree(node) > 0 and nx_graph.out_degree(node) == 0:
                highest_conclusions.append(str(node))
        return highest_conclusions

    def get_pred_height(self):
        nx_graph = self.get_pred_nx_graph()

        # if nx.is_directed_acyclic_graph(nx_graph):
        #     return nx.dag_longest_path_length(nx_graph)
        # else:
        #     print("predicted graph is not DAG")
        #     return 1e2

        try:
            return nx.dag_longest_path_length(nx_graph)
        except Exception as e:
            print(e)
            return 1e2

    def __str__(self):
        s = ""
        s += f"H: {self.H}\n"
        
        s += f"S:\n"
        for sent in self.S:
            s += f"\t{self.sent2id[sent]}:  {sent}\n"

        s += f"used S:\n"
        for sent in self.used_S:
            s += f"\t{self.sent2id[sent]}:  {sent}\n"
            
        s += f"P:\n"
        for step in self.P:
            s += f"{' & '.join([self.sent2id[p] for p in step['pre_sent']])} -> {self.sent2id[step['con_sent']]}; "
    
        # s += f"\nQ: {self.Q}\n" 
        # s += f"A: {self.A}\n"  
        return s 

    def next_id(self, ident='int'):
        assert ident in ['sent', 'int']
        for i in itertools.count(1):
            if f"{ident}{i}" not in self.sent2id.values():
                return f"{ident}{i}"  
            
    def reassign_sent2id(self):
        """
        reassign the identifier of sent and int
        sent2id contain sentences in S / used_S / H
        """
        sent_count = 1
        int_count = 1
        
        for sent in self.used_S:
            if self.sent2id[sent].startswith('sent'):
                self.sent2id[sent] = f"sent{sent_count}"
                sent_count += 1
            elif self.sent2id[sent].startswith('int'):
                self.sent2id[sent] = f"int{int_count}"
                int_count += 1
            
        for sent in self.S:
            if self.sent2id[sent].startswith('sent'):
                self.sent2id[sent] = f"sent{sent_count}"
                sent_count += 1
            elif self.sent2id[sent].startswith('int'):
                self.sent2id[sent] = f"int{int_count}"
                int_count += 1          
        
    @classmethod
    def P_to_nx_graph(cls, P, sent2id):
        """
        convert the partial proof to nx-graph
        """

        G = nx.DiGraph()
        for step in P:
            # add node
            for sent in step['pre_sent']+[step['con_sent']]:
                G.add_node(sent, id=sent2id[sent], sent=sent)
            # add edge
            for sent in step['pre_sent']:
                G.add_edge(sent, step['con_sent'])
        return G

    def linearize_proof(self):
        proof_str = ""
        for step in self.P:
            proof_str += " & ".join(sorted([self.sent2id[sent] for sent in step['pre_sent']]))
            proof_str += f" -> {self.sent2id[step['con_sent']]}; "
        proof_str = proof_str[:-2]
        return proof_str
    
    def linearize_context(self, sent_list = None):
        s = ""
        # for sent in sent_list:
        #     s += f"{self.sent2id[sent]}: {add_fullstop(sent)} "

        sentX_sents = []
        not_sentX_sents = []
        for sent in sent_list:
            if self.sent2id[sent].startswith('sent'):
                sentX_sents.append(sent)
            else:
                not_sentX_sents.append(sent)

        for sent in not_sentX_sents + sentX_sents:
            s += f"{self.sent2id[sent]}: {add_fullstop(sent)} "
        
        return s
    
    def linearize_state(self, form = 'default'):
        
        if form == 'default' or form == 'HPS':
            state_str = ""
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$context$ {self.linearize_context(self.S)} "
        elif form == 'QAHPS':
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$context$ {self.linearize_context(self.S)} " 
        elif form == 'QAHPN':
            node_sents = []
            for step in self.P[::-1]: #  from root to leaves
                for sent in [step['con_sent']] + step['pre_sent']:
                    if sent not in node_sents:
                        node_sents.append(sent)
                        
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$node$ {self.linearize_context(node_sents)} "     
        elif form == 'QACHPN':
            node_sents = []
            for step in self.P[::-1]: #  from root to leaves
                for sent in [step['con_sent']] + step['pre_sent']:
                    if sent not in node_sents:
                        node_sents.append(sent)
                        
            state_str = ""
            state_str += f"$question$ {self.Q} $answer$ {add_fullstop(self.A.lower())} "
            state_str += f"$choices$ {self.choices_str} "
            state_str += f"$hypothesis$ {add_fullstop(self.H)} "
            state_str += f"$proof$ {self.linearize_proof()} "
            state_str += f"$node$ {self.linearize_context(node_sents)} "     
        else:
            # state_str += f"$used$ {self.linearize_context(self.used_S)}"
            raise NotImplementedError
        
        return state_str

    def to_dict(self):
        return copy.deepcopy({
            'H': self.H,
            'Q': self.Q,
            'A': self.A,
            'S': self.S,
            'used_S': self.used_S,
            'P': self.P,
            'sent2id': self.sent2id,
            'history_actions': self.history_actions,
            'choices_str': self.choices_str,
        })

# class MyEnv(gym.Env): # ref: https://www.gymlibrary.ml/content/environment_creation/
class EntailmentTreeEnv:
    def __init__(self,  retriever = None, 
                        entailment_module = None, 
                        verifier = None,
                        env_args = {}
                        ):
        """
        
        """
        self.retriever = retriever
        self.entailment_module = entailment_module
        self.verifier = verifier

        self.env_args = env_args
        self.action_budget = env_args.get('action_budget', {})

        self.retrieve_top_n = env_args.get('retrieve_top_n', 25)
        self.max_height = env_args.get('max_height', 10)
        self.check_premise_overlap = bool(env_args.get('check_premise_overlap', 0))
        self.step_score_thre = env_args.get('step_score_thre', 0.0)
        self.module_num_return = env_args.get('module_num_return', 5)


        self.state = None
        self.history_actions = []
        self.history_trace = []
        self.done = False

        # ! remember to check the self.copy_env function while changing arguments

    def reset(self, data_item = None):
        """
        The reset method will be called to initiate a new episode. 
        """
        # super().reset(seed=seed)
        self.state = State(data_item)
        self.history_actions = []
        self.history_trace = []
        self.done = False
        
        
    def step(self, action):
        """
        The step method accepts an action, computes the state of the environment 
        after applying that action and returns the 4-tuple (observation, reward, done, info)
            observation (object): state
            reward (float)
            done (boolen)
            info: other helpful infomation
        
        action (dict): {'type':'retrieve', 'query':'the sun', ...}
        """
        
        if self.done:
            return self.state, 0.0, self.done, []

        if self.check_action_executable(action) == False:
            # print(f"action is not executable: {action}")
            self.done = True
            return self.state, 0.0, self.done, []

        state_before_action = self.state.to_dict()

        # ----- execute action ----- 
        action_failed = False
        if action['type'] == Action.retrieve:
            """
            retrieve with action['query'] and update the state S with retrieval result + query
            If a query is consecutively queried k times, we scroll down the retrieval result k times
            """

            if 'query' in action:
                query = action['query']
            else:
                id2sent = {sent_id:sent for sent, sent_id in self.state.sent2id.items()}
                query = id2sent.get(action['query_id'], "")
                action['query'] = query
                
            if not query:
                action_failed = True
            else:
                repeat_time = 0
                for ac in self.history_actions[::-1]:
                    if ac['type'] == Action.retrieve:
                        if ac['query'] == query:
                            repeat_time += 1
                            # print(repeat_time, query)
                        else:
                            pass
                    else:
                        break
                
                if repeat_time == 0:
                    retrieval_result = self.retriever(query, n = self.retrieve_top_n)
                else:
                    retrieval_result = self.retriever(query, n = self.retrieve_top_n*(repeat_time+1))
                    retrieval_result = retrieval_result[self.retrieve_top_n*repeat_time: self.retrieve_top_n*(repeat_time+1)]

                retrieval_result = [r['text'] for r in retrieval_result]
                if query != self.state.H:
                    retrieval_result = [query] + retrieval_result
                self.state.update_S(retrieval_result)
                

        elif action['type'] == Action.reason:
            if 'pre_sent' in action['step']:
                pre_sent = action['step']['pre_sent']
            else:
                id2sent = {sent_id:sent for sent, sent_id in self.state.sent2id.items()}
                pre_sent = [id2sent.get(p, "") for p in action['step']['pre_id']]
                action['step']['pre_sent'] = pre_sent
            
            if action.get('use_module', False) == True:
                # use entailment module in the env to reason a step
                action['step'] = self.reason_entailment_module_steps([action['step']])[0]
                
                # filter steps
                if action['step']['step_scorer_score'] < self.step_score_thre:
                    # print(f"Filter step with score: {self.step_score_thre}")
                    # print(action['step'])
                    action_failed = True


            assert 'con_sent' in action['step']

            if action['step']['con_sent'] in self.state.S:
                if action['step']['con_sent'] != self.state.H:
                    del self.state.sent2id[action['step']['con_sent']]

            self.state.expand_P(action['step'])

        elif action['type'] == Action.end:
            # print("done: action end")
            self.done = True

        else:
            raise NotImplementedError

        # ----- check action_failed -----
        if action_failed:
            # print(f"done: action failed! {action}")
            self.done = True
            # return self.state, 0.0, self.done, []

        # ----- add history actions -----
        self.history_trace.append((copy.deepcopy(state_before_action), copy.deepcopy(action)))
        self.history_actions.append(copy.deepcopy(action))  

        # print(State(state_before_action).linearize_state())
        # print(action)
        # print("*"*20)

        self.state.history_actions = self.history_actions
        
        # ----- check action budget -----
        for action_type, action_budge in self.action_budget.items():
            if sum([ac['type']==action_type for ac in self.history_actions]) >= action_budge:
                # print(f"done: {action_type} reach budget {action_budge}")
                self.done = True

        # ----- check proved -----
        # Sometimes H can be retrieved directly
        if any([same_sent(sent, self.state.H) for sent in self.state.S]):
            # print("done: H in S")
            self.done = True
        elif any([same_sent(step['con_sent'], self.state.H) for step in self.state.P]):
            # print("done: H in P")
            self.done = True

        # ----- check predicted tree height -----
        if self.state.get_pred_height() >= self.max_height:
            # print(f"done: Reach maximal height {self.state.get_pred_height()} >= {self.max_height}")
            self.done = True

        # ----- check S and P for task 1 -----
        if len(self.state.P) > 0 and len(self.state.S) < 2:
            # print(f"done: length of S {len(self.state.S)}")
            self.done = True

        observation = self.state
        reward = 0.0
        done = self.done
        info = []

        return observation, reward, done, info

    def render(self):
        pass

    def check_action_executable(self, action_str):
        if action_str is None:
            return False

        # try to parse the str
        if type(action_str) == str:
            try:
                action = Action.parse_action(action_str)
            except Exception as e:
                # print(f"check_action_executable failed: {str(e)}. action_str: {action_str}")
                action = None

            if action is None:
                return False
        else:
            action = action_str
            if 'type' not in action:
                return False

        # check by action type
        if action['type'] == Action.retrieve:
            if 'query' in action:
                query = action['query']
            elif 'query_id' in action:
                id2sent = {sent_id:sent for sent, sent_id in self.state.sent2id.items()}
                query = id2sent.get(action['query_id'], "")
                action['query'] = query
            else:
                return False

            if not query:
                return False

        elif action['type'] == Action.reason:
            if 'step' not in action:
                return False

            # check 'pre_id' / 'pre_sent'
            if 'pre_sent' in action['step']:
                pre_sent = action['step']['pre_sent']
            elif 'pre_id' in action['step']:
                id2sent = {sent_id:sent for sent, sent_id in self.state.sent2id.items()}
                pre_sent = [id2sent.get(p, "") for p in action['step']['pre_id']]
                action['step']['pre_sent'] = pre_sent
            else:
                return False

            if any([sent not in self.state.S for sent in pre_sent]):
                # premise not in state.S
                return False
            if len(set(action['step']['pre_sent'])) != len(action['step']['pre_sent']):
                # premises: sent1 & sent1
                return False

            # filter pre_sent by rules
            if self.check_premise_overlap:
                # Rule: if the pre_sent have no overlap or have too much overlap, we reject the step            
                if len(action['step']['pre_sent']) < 2:
                    return False
                pre_iou = max([sent_IoU(ps[0], ps[1], spacy_nlp) for ps in itertools.combinations(action['step']['pre_sent'], 2)])
                if pre_iou == 0.0 or pre_iou >= 0.7:
                    # print("*** filter step ***", action['step']['pre_sent'])
                    return False
            

            # check 'con_sent'
            if 'con_sent' not in action['step']:
                # use module to get con_sent
                pass
            else:
                con_sent = action['step']['con_sent']
                if any([same_sent(con_sent, s) for s in pre_sent]):
                    # conclusion repeats one of the premises 
                    return False
                if any([same_sent(con_sent, s) for s in self.state.used_S]):
                    # print('con_sent in used_S')
                    return False
                
                S_int = [sent for sent in self.state.S if self.state.sent2id[sent].startswith('int')]
                if any([same_sent(con_sent, s) for s in S_int]):
                    # some intermediate conclusions could be found as facts in corpus
                    # we only return false when the con_sent has been a int
                    # print('con_sent has been int')
                    return False


        elif action['type'] == Action.end:
            pass

        else:
            raise NotImplementedError
            
        return True

    def get_score_compare_gold(self, bleurt_scorer):

        def is_step_pre_in_S(step, S):
            return all([p in S for p in step['pre_sent']])

        state = self.state
        score = {}

        # ----- gold info -----
        gold_node_list = get_gt_node_list(state.data_item)
        id2sent = {node['id']:node['sent'] for node in gold_node_list}
        gold_sents = [node['sent'] for node in gold_node_list if node['id'].startswith('sent')]
        gold_steps = [{'pre':sorted(node['pre']), 'pre_sent': sorted([id2sent[p] for p in node['pre']]),
                    'con':node['id'], 'con_sent': id2sent[node['id']]}
                    for node in gold_node_list 
                    if len(node['pre'])]

        gold_next_steps = []
        for step in gold_steps:
            if is_step_pre_in_S(step, state.S):
                gold_next_steps.append(step)

        # ----- S -----
        # used_sent = sum([step['pre_sent'] for step in state.P],[])
        # unused_gold_sents = set(gold_sents) - set(used_sent) 
        # if len(unused_gold_sents) == 0: # bug here? for int1 & int2
        #     recall = -1.0
        # else:
        #     recall = sum([sent in state.S for sent in unused_gold_sents])/(len(unused_gold_sents)+1e-10)
        # score['S_recall'] = recall

        recall = sum([sent in (state.S + state.used_S) for sent in gold_sents])/(len(gold_sents)+1e-20)
        score['S_recall'] = recall

        score['S_contain_step'] = len(gold_next_steps)  

        # ----- P (tree) -----
        pred_node_list = state.get_pred_node_list()
        eval_result = eval_tree_task3(pred_node_list, gold_node_list, bleurt_scorer)
        score['tree_score'] = eval_result

        return score

    def reason_entailment_module_steps(self, steps):

        # module generate parameters
        prefixes = [
            'deductive substitution:',
            'deductive conjunction:',
            'deductive if-then:',
        ]
        mode = 'beam_search'
        generate_args = {
            'num_beams' : self.module_num_return,
            'num_return_sequences': self.module_num_return,
        }

        # module inference
        input_sents = []
        for step in steps:
            input_sents.append(self.entailment_module.make_input_sent(step['pre_sent'], 
                                        H=self.state.H, Q=self.state.Q, A=self.state.A))

        module_preds, module_generate_scores, input_infos = self.entailment_module(input_sents=input_sents, prefixes=prefixes, 
                                                                         mode = mode, generate_args=generate_args)

        # filter by rule
        S_int = [sent for sent in self.state.S if self.state.sent2id[sent].startswith('int')]
        for idx, input_sent in enumerate(input_sents):
            for pred_idx in range(len(module_preds[idx])):
                # see self.check_action_executable
                forbidden_cons = steps[idx]['pre_sent'] + self.state.used_S + S_int
                if any([same_sent(module_preds[idx][pred_idx], s) for s in forbidden_cons]):
                    module_generate_scores[idx][pred_idx] = -1

        # verify and select the best conclusion sentence
        for idx in range(len(steps)):
            pre_sent = steps[idx]['pre_sent']
            
            preds_ = module_preds[idx]
            generate_scores_ = module_generate_scores[idx]
            module_inputs_ = input_infos[idx]
        
            verifier_inputs = [
                    {'pre_sent': pre_sent, 'con_sent': con_sent}
                    for con_sent in preds_
            ]
            step_scorer_scores_ = self.verifier.verify_steps(verifier_inputs)

            # merge_scores_ = 0.5*np.array(generate_scores_) + 0.5*np.array(verifier_scores_)
            merge_scores_ = np.array(generate_scores_)*np.array(step_scorer_scores_)

            max_i = np.argmax(merge_scores_)
            
            steps[idx]['con_sent'] = preds_[max_i]
            steps[idx]['generate_score'] = generate_scores_[max_i]
            steps[idx]['step_scorer_score'] = step_scorer_scores_[max_i]
            steps[idx]['merge_score'] = merge_scores_[max_i]
            steps[idx]['module_input'] = module_inputs_[max_i]

        return steps

    def copy_env(self):

        new_env = EntailmentTreeEnv(retriever = self.retriever,
                                    entailment_module = self.entailment_module,
                                    verifier = self.verifier,
                                    env_args = self.env_args,
                                    )

        new_env.state = copy.deepcopy(self.state)
        new_env.history_actions = copy.deepcopy(self.history_actions)
        new_env.history_trace = copy.deepcopy(self.history_trace)
        new_env.done = self.done

        return new_env