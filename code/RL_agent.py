from tree_utils import *
import numpy as np
import random
from RL_env import Action, State, EntailmentTreeEnv
import math


def oracle_strategy_next_action(state, retriever, retrieve_top_n = 25):
    """
    perfect reasoner +  unperfect retriever
    """
    
    def is_step_pre_in_S(step, S, assert_pre=None):
        if assert_pre is None:
            return all([p in S for p in step['pre_sent']])
        else:
            return all([p in S for p in step['pre_sent']]) and (assert_pre in step['pre_sent'])

    # collect gold infomation
    gold_nx_graph = get_gt_nx_graph(state.data_item)

    gold_node_list = get_gt_node_list(state.data_item)
    id2sent = {node['id']:node['sent'] for node in gold_node_list}
    gold_sents = [node['sent'] for node in gold_node_list if node['id'].startswith('sent')]
    gold_steps = [{'pre':sorted(node['pre']), 'pre_sent': sorted([id2sent[p] for p in node['pre']]),
                   'con':node['id'], 'con_sent': id2sent[node['id']]}
                 for node in gold_node_list 
                 if len(node['pre'])]


    # Here, we assert that the partial proof are correct

    # if hypothesis in S or P, the next action is end
    # if any([same_sent(step['con_sent'], state.H) for step in state.P]):
    #     next_action = {'type': Action.end, 'is_proved': 'proved'}
    #     return next_action
    if any([same_sent(s, state.H) for s in state.S]):
        next_action = {'type': Action.end, 'is_proved': 'proved'}
        return next_action

    # if S contain one gold step, the next action is this step
    gold_next_steps = []
    for step in gold_steps:
        if is_step_pre_in_S(step, state.S):
            gold_next_steps.append(step)
            break

    if len(gold_next_steps) > 0:
        # selected_step = random.choice(gold_next_steps)
        selected_step = gold_next_steps[-1]
        next_action = {
            'type': Action.reason,
            'step': {
                'pre_sent': selected_step['pre_sent'],
                'con_sent': selected_step['con_sent'],
                'pre_id': [state.sent2id[sent] for sent in selected_step['pre_sent']],
            } 
        }
        return next_action

    # if S do not contain any gold step, the next action is retrieval, we should find the best query
    # priority1: the retrieval result cantain one gold step (with query); score 100
    # priority2: the retrieval result cantain one gold step (without query); score 10
    # priority3: the retrieval result cantain more unused gold sents; score 1~0

    candidate_query = [state.H] + state.S
    # candidate_query = [state.H]
    # for s in state.S:
    #     if state.sent2id[s].startswith('int'):
    #         candidate_query.append(s)

    candidate_query_priority = []

    for query in candidate_query:
        new_S = []

        if query != state.H:
            new_S.append(query)

        # If a query is consecutively queried k times, we scroll down the retrieval result k times
        repeat_time = 0
        for ac in state.history_actions[::-1]:
            if ac['type'] == Action.retrieve:
                if ac['query'] == query:
                    repeat_time += 1
            else:
                break

        if repeat_time == 0:
            retrieval_result = retriever(query, n = retrieve_top_n)
        else:
            retrieval_result = retriever(query, n = retrieve_top_n*(repeat_time+1))
            retrieval_result = retrieval_result[retrieve_top_n*(repeat_time): retrieve_top_n*(repeat_time+1)]

        for r in retrieval_result:
            if r['text'] not in state.used_S:
                new_S.append(r['text'])

        if any([is_step_pre_in_S(step, new_S, assert_pre=query) for step in gold_steps]):
            candidate_query_priority.append(100)
        elif any([is_step_pre_in_S(step, new_S) for step in gold_steps]):
            candidate_query_priority.append(10)
        else:
            used_sent = sum([step['pre_sent'] for step in state.P],[])
            unused_gold_sents = set(gold_sents) - set(used_sent)
            recall = sum([sent in new_S for sent in unused_gold_sents])/(len(unused_gold_sents)+1e-10)
            candidate_query_priority.append(recall)

    selected_query = candidate_query[np.argmax(candidate_query_priority)]
    next_action = {
        'type': Action.retrieve,
        'query': selected_query,
        'query_id': state.sent2id[selected_query],
        'query_priority':float(np.max(candidate_query_priority)),
    }
    return next_action


def interact(entailment_env, controller, interact_args = {}, verbose = False):

    linearize_state_form = interact_args['linearize_state_form'] # input form of the controller
    generate_args = interact_args['controller_generate_args'] # generate_args of controller
    constraints_args_reason = interact_args['constraints_args_reason'] # constraint `reason` generate_args of controller
    
    force_reason_strategy = interact_args.get('force_reason_strategy', None)
    action_score_strategy = interact_args.get('action_score_strategy', None)

    lookahead_L = interact_args.get('lookahead_L', 0)
    
    excuted_action_num = 0
    while True:
        entailment_env.state.reassign_sent2id()

        # -------- get next action from controller --------
        controller_inputs = [entailment_env.state.linearize_state(form = linearize_state_form)]
        
        constrain_reason = False
        if force_reason_strategy in ['None', None]:
            pass
        elif force_reason_strategy == 'after_retrieve':
            if len(entailment_env.history_actions) == 0 or entailment_env.history_actions[-1]['type'] == Action.retrieve:
                constrain_reason = True
        elif force_reason_strategy == 'always':
            constrain_reason = True
        else:
            raise NotImplemented
            
        controller_preds = None
        if constrain_reason:
            controller_preds = controller(controller_inputs, mode='constrained_beam_search', 
                                          generate_args=constraints_args_reason)[0]
            action_strs = controller_preds['text']
            if not any([entailment_env.check_action_executable(ac) for ac in action_strs]): 
                # No executable action for constrained generation
                controller_preds = None

        # normal predict
        if controller_preds is None:
            controller_preds = controller(controller_inputs, mode='beam_search', generate_args=generate_args)[0]
        
        action_strs, controller_generate_scores, controller_first_token_scores = \
            controller_preds['text'], controller_preds['sequence_score'], controller_preds['first_token_score']
            
        
        # -------- score the actions --------
        executable_index = [idx for idx in range(len(action_strs)) 
                            if entailment_env.check_action_executable(action_strs[idx])]
        if len(executable_index) == 0:
            # print("No executable action")
            break

        actions = []
        for idx in executable_index:
            action = Action.parse_action(action_strs[idx])
            action['controller_generate_score'] = controller_generate_scores[idx]
            action['controller_first_token_score'] = controller_first_token_scores[idx]
            actions.append(action)
            
        if action_score_strategy in ['None', None, 'only_controller']:
            for action in actions:
                action['merged_score'] = action['controller_generate_score']
            excuted_action_num += 1

        elif action_score_strategy in ['lookahead', 'lookahead_only_state', 'lookahead_v2']:

            if lookahead_L == 0:
                for action in actions:
                    lookahead_env = entailment_env.copy_env()
                    state, _, done,_ = lookahead_env.step(action)
                    state_score = entailment_env.verifier.verify_a_state(lookahead_env.state)
                    
                    if action_score_strategy in ['lookahead', 'lookahead_v2']:
                        action['merged_score'] = state_score + action['controller_generate_score']
                    elif action_score_strategy == 'lookahead_only_state':
                        action['merged_score'] = state_score
                    else:
                        raise NotImplemented

                excuted_action_num += len(actions)

            else:
                raise NotImplemented
        
        else:
            raise NotImplemented
        
        sorted_actions = sorted(actions, key=lambda x:x['merged_score'], reverse=True)
        best_action = sorted_actions[0]
    
        # print(Action.linearize_action(best_action))

        # -------- excute the best action --------
        state,_, done,_ = entailment_env.step(best_action)
        
        if done:
            # print('done')
            break
            

    extra_info = {
        'excuted_action_num': excuted_action_num,
    }

    return entailment_env, extra_info 


### ----- MCTS -----
### ref: https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py 
class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class MCTS_Node:
    def __init__(self, node_env, prior, node_action = None):
        self.node_env = node_env
        self.prior = prior
        self.node_action = node_action # the detailed action from parent to this node
        
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        
        self.terminal = False # is this node or all its children are done
        
        self.depth_to_leaf = 0
        
        self.node_score = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    
    def expand(self, reward, node_score, candidate_actions):
        # set the reward of this node
        self.reward = reward
        self.node_score = node_score
        
        # expand the children
        if len(candidate_actions) == 0:
            self.terminal = True
        
        for action in candidate_actions:
            action_str = Action.linearize_action(action)
            self.children[action_str] = MCTS_Node(node_env = self.node_env.copy_env(), 
                                                  prior = action['controller_generate_score'],
                                                  node_action = action)
        

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS_Searcher:

    def __init__(self, interact_args):


        self.linearize_state_form = interact_args['linearize_state_form'] # input form of the controller
        self.generate_args = interact_args['controller_generate_args'] # generate_args of controller
        self.constraints_args_reason = interact_args['constraints_args_reason'] # constraint `reason` generate_args of controller

        self.force_reason_strategy = interact_args.get('force_reason_strategy', None)
        
        self.puct = interact_args.get('puct', 1.0)
        self.discount = interact_args.get('discount', 1.0)

        # print(f"-------- MCTS Searcher puct: {self.puct} --------")
        
        
    def run(
        self,
        entailment_env,
        controller,
        num_simulations,
        verbose = False
    ):
        
        # initialize root note
        root = MCTS_Node(entailment_env, prior = 0)
        candidate_actions = self.get_candidate_actions(root.node_env, controller)
        if len(candidate_actions) == 0:
            return root.node_env, {"max_tree_depth": 0,"search_root": root, 'excuted_action_num': 0}
        root.expand(reward = 0, node_score = 0, candidate_actions = candidate_actions)
        root.visit_count = 1
        
        # repeat 
        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        excuted_action_num = 0
        for sim_t in range(num_simulations):
            # print('min_max_stats', min_max_stats.maximum, min_max_stats.minimum)
            
            node = root
            search_path = [node]
            current_tree_depth = 0
            
            # if node.expanded():
            #     for ac, child in node.children.items():
            #         print(sim_t, ac, round(self.ucb_score(node, child, min_max_stats), 2), child.visit_count)
                

            # ----- Selection -----
            while node.expanded():
                current_tree_depth += 1
                _, node = self.select_child(node, min_max_stats, exclude_terminal = True)
                if node is None: break
                search_path.append(node)
                
            if search_path[-1].expanded(): break # the whole search space have done

                
            # print(f"Selection: {search_path[-1].node_env.state.linearize_proof()}")
            # print(' -> '.join([ac['type'] for ac in node.node_env.history_actions]))
                
            if verbose:
                print(f"Selection: {search_path[-1].node_env.state.linearize_proof()}")
                print(' -> '.join([ac['type'] for ac in node.node_env.history_actions]))
                print(f"Selection: {self.ucb_score(search_path[-2], search_path[-1], min_max_stats)}")
            

            # ----- Expansion -----
            # execute the action
            leaf_node = search_path[-1]
            action = leaf_node.node_action
            
            _, _, done,_ = leaf_node.node_env.step(action) # TODO: return reward from env

            if verbose:
                print(f"Expansion: {Action.linearize_action(action)}")

            # compute the reward
            node_score = leaf_node.node_env.verifier.verify_a_state(leaf_node.node_env.state)
            reward = None
    
            # predict next_actions and expand the node
            if done:
                # the node will be terminal
                candidate_actions = [] 
            else:
                candidate_actions = self.get_candidate_actions(leaf_node.node_env, controller)

            leaf_node.expand(reward = 0.0, node_score = node_score, candidate_actions = candidate_actions)

            # ----- Backpropagation -----
            self.backpropagate(search_path, node_score, min_max_stats) # TODO: check the value here

            max_tree_depth = max(max_tree_depth, current_tree_depth)
            excuted_action_num += 1
        
        # select a final env and return
        best_node = self.select_best_node(root)
        # print(f"Final node terminal {best_node.terminal}")
        
        # extra_info
        extra_info = {
            "max_tree_depth": max_tree_depth,
            "search_root": root, 
            "excuted_action_num": excuted_action_num,
        }
        
        return best_node.node_env, extra_info

    def select_child(self, node, min_max_stats, exclude_terminal = False):
        """
        Select the child with the highest UCB score.
        """
        
        ### be careful to think about exclude_terminal when selecting child; it could make the MCTS invalid (depend on the Q)
        if exclude_terminal:
            not_terminal_children = [child for action, child in node.children.items() if child.terminal == False]
            if len(not_terminal_children) == 0: return None, None
            max_ucb = max([self.ucb_score(node, child, min_max_stats) for child in not_terminal_children])
        else:
            max_ucb = max([self.ucb_score(node, child, min_max_stats) for action, child in node.children.items()])
        
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        ### MuZero UCB
        # pb_c = (
        #     math.log(
        #         (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
        #     )
        #     + self.config.pb_c_init
        # )
        # pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        ### AlphaZero UCB
        pb_c = self.puct * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(self.Q_func(child))
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
        """
        for node in reversed(search_path):
            # update value
            if node.visit_count == 0:
                # the leaf node in the path, its initail value is 0.0
                value_to_add = value
            else:
                if len([child.value() for _, child in node.children.items()]) == 0:
                    # print(node.visit_count, node.children.keys())
                    # print(search_path)
                    value_to_add = value
                else:
                    value_to_add = np.max([child.value() for _, child in node.children.items()])
            
            node.visit_count += 1
            node.value_sum += value_to_add
            
            min_max_stats.update(self.Q_func(node))

            # value = node.reward + self.discount * value

            # update node terminal
            if node.terminal == False:
                if all([child.terminal for _, child in node.children.items()]):
                    node.terminal = True
                    
            # update depth_to_leaf
            if node.expanded():
                node.depth_to_leaf = np.max([child.depth_to_leaf for _, child in node.children.items()]) + 1

    def get_candidate_actions(self, input_env, controller):
        # -------- get next action from controller --------
        linearize_state_form = self.linearize_state_form
        force_reason_strategy = self.force_reason_strategy
        
        generate_args = self.generate_args
        constraints_args_reason = self.constraints_args_reason
        
        # reassign_sent2id
        input_env.state.reassign_sent2id()

        controller_inputs = [input_env.state.linearize_state(form = linearize_state_form)]

        constrain_reason = False
        if force_reason_strategy in ['None', None]:
            pass
        elif force_reason_strategy == 'after_retrieve':
            if len(input_env.history_actions) == 0 or input_env.history_actions[-1]['type'] == Action.retrieve:
                constrain_reason = True
        elif force_reason_strategy == 'always':
            constrain_reason = True
        else:
            raise NotImplemented

        controller_preds = None
        if constrain_reason:
            controller_preds = controller(controller_inputs, mode='constrained_beam_search', 
                                          generate_args=constraints_args_reason)[0]
            action_strs = controller_preds['text']
            if not any([input_env.check_action_executable(ac) for ac in action_strs]): 
                # No executable action for constrained generation
                controller_preds = None

        # normal predict
        if controller_preds is None:
            controller_preds = controller(controller_inputs, mode='beam_search', generate_args=generate_args)[0]

        action_strs, controller_generate_scores, controller_first_token_scores = \
            controller_preds['text'], controller_preds['sequence_score'], controller_preds['first_token_score']


        # -------- score the actions --------
        executable_index = [idx for idx in range(len(action_strs)) 
                            if input_env.check_action_executable(action_strs[idx])]
        if len(executable_index) == 0:
            # print("No executable action")
            return []

        actions = []
        for idx in executable_index:
            action = Action.parse_action(action_strs[idx])
            action['controller_generate_score'] = controller_generate_scores[idx]
            action['controller_first_token_score'] = controller_first_token_scores[idx]
            actions.append(action)
            
        return actions
   
    def Q_func(self, node):
        return node.value()

    def qp_score(self, child):
        prior_score = self.puct * child.prior
        value_score = self.Q_func(child) if child.visit_count > 0 else 0
        return prior_score + value_score

    def select_child_qp_score(self, node):
        max_ucb = max([self.qp_score(child) for action, child in node.children.items()])
        action = [action for action, child in node.children.items() if self.qp_score(child) == max_ucb][0]
        return action, node.children[action]
        
    def select_best_node(self, root):
        node = root
        search_path = [node]
        current_tree_depth = 0

        while node.expanded():
            current_tree_depth += 1
            _, node = self.select_child_qp_score(node)
            search_path.append(node)
            
        best_node = search_path[-1]
        return best_node


class Beam_Searcher:

    def __init__(self, interact_args):


        self.linearize_state_form = interact_args['linearize_state_form'] # input form of the controller
        self.generate_args = interact_args['controller_generate_args'] # generate_args of controller
        self.constraints_args_reason = interact_args['constraints_args_reason'] # constraint `reason` generate_args of controller

        self.force_reason_strategy = interact_args.get('force_reason_strategy', None)
        
        self.beam_size = interact_args.get('beam_size', 1)

        # print(f"-------- Beam Searcher beam_size: {self.beam_size} --------")
        
        
    def run(
        self,
        entailment_env,
        controller,
        num_simulations,
        verbose = False
    ):
            
        excuted_action_num = 0

        # initial beams
        beams = [entailment_env.copy_env()]
        beams_scores = [0.0]
            
        beam_step = 0
        while True:
            
            next_envs = []
            next_envs_scores = []
            for beam_env, beam_env_score in zip(beams, beams_scores):
                beam_env.state.reassign_sent2id()
                
                if beam_env.done:
                    # the state is done, no more actions
                    next_envs.append(beam_env)
                    next_envs_scores.append(beam_env_score)
                    
                else:
                    # excute candidate actions to get next state
                    actions = self.get_candidate_actions(beam_env, controller)

                    for action in actions:
                        next_env = beam_env.copy_env()
                        state, _, done,_ = next_env.step(action)
                        state_score = entailment_env.verifier.verify_a_state(next_env.state)

                        action['merged_score'] = state_score + action['controller_generate_score']

                        next_envs.append(next_env)
                        next_envs_scores.append(state_score) # only state score

                        # print(Action.linearize_action(action), state_score)

                    excuted_action_num += len(actions)
                    if excuted_action_num > num_simulations:
                        break

                
            # print(next_envs_scores)


            # select the top-k env for next step search
            if len(next_envs) == 0 or all([next_env.done for next_env in next_envs]):
                break
            else:
                beams = []
                beams_scores = []
                sorted_idx = np.argsort(next_envs_scores)[::-1] # from large to small
                for idx in sorted_idx[:self.beam_size]:
                    beams.append(next_envs[idx])
                    beams_scores.append(next_envs_scores[idx])
                    
                beam_step += 1
                    
                # print(f"beam_step:{beam_step} \t beams_scores: {beams_scores}")
    
            if excuted_action_num > num_simulations:
                break
                
        # print("Beam search done")
        best_env = beams[np.argmax(beams_scores)]
    
        # extra_info
        extra_info = {
            "excuted_action_num": excuted_action_num,
        }
        
        return best_env, extra_info

    def get_candidate_actions(self, input_env, controller):
        # -------- get next action from controller --------
        linearize_state_form = self.linearize_state_form
        force_reason_strategy = self.force_reason_strategy
        
        generate_args = self.generate_args
        constraints_args_reason = self.constraints_args_reason
        
        # reassign_sent2id
        input_env.state.reassign_sent2id()

        controller_inputs = [input_env.state.linearize_state(form = linearize_state_form)]

        constrain_reason = False
        if force_reason_strategy in ['None', None]:
            pass
        elif force_reason_strategy == 'after_retrieve':
            if len(input_env.history_actions) == 0 or input_env.history_actions[-1]['type'] == Action.retrieve:
                constrain_reason = True
        elif force_reason_strategy == 'always':
            constrain_reason = True
        else:
            raise NotImplemented

        controller_preds = None
        if constrain_reason:
            controller_preds = controller(controller_inputs, mode='constrained_beam_search', 
                                          generate_args=constraints_args_reason)[0]
            action_strs = controller_preds['text']
            if not any([input_env.check_action_executable(ac) for ac in action_strs]): 
                # No executable action for constrained generation
                controller_preds = None

        # normal predict
        if controller_preds is None:
            controller_preds = controller(controller_inputs, mode='beam_search', generate_args=generate_args)[0]

        action_strs, controller_generate_scores, controller_first_token_scores = \
            controller_preds['text'], controller_preds['sequence_score'], controller_preds['first_token_score']


        # -------- score the actions --------
        executable_index = [idx for idx in range(len(action_strs)) 
                            if input_env.check_action_executable(action_strs[idx])]
        if len(executable_index) == 0:
            # print("No executable action")
            return []

        # print("action_strs", action_strs)

        actions = []
        for idx in executable_index:
            action = Action.parse_action(action_strs[idx])
            action['controller_generate_score'] = controller_generate_scores[idx]
            action['controller_first_token_score'] = controller_first_token_scores[idx]

            # if action not in actions:
            actions.append(action)
            
        return actions
