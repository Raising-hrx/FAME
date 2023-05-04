import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import math
import time
import argparse
from collections import defaultdict, Counter
from itertools import combinations
import itertools

import numpy as np
import torch
import transformers


from tree_utils import *
from RL_env import EntailmentTreeEnv, State, Action

from StepScorer import load_step_scorer, predict_step_scorer, StepScorer


class Verifier:
    def __init__(self, step_scorer = None, entail_scorer = None, bleurt_scorer= None,
                P_score_type = None, H_score_type = None, merge_strategy = None):
        
        self.step_scorer = step_scorer
        self.entail_scorer = entail_scorer
        self.bleurt_scorer = bleurt_scorer

        self.P_score_type = P_score_type
        self.H_score_type = H_score_type
        self.merge_strategy = merge_strategy


    def __call__(self):
        pass
    
    def verify_a_state(self, state, return_dict = False):
        highest_conclusions = state.get_pred_highest_conclusions()

        verifer_info = {}

        # ----- score of P -----
        if len(state.P) == 0 or len(highest_conclusions) == 0:
            P_score = 0.0
        else:
            # get step score from step scorer
            steps_index = []
            steps = []
            for si, step in enumerate(state.P):
                if 'step_scorer_score' not in step.keys():
                    steps_index.append(si)
                    steps.append({'pre_sent': step['pre_sent'], 
                                  'con_sent': step['con_sent'],})

            if steps:
                step_scores = self.step_scorer(steps)
                for score, index in zip(step_scores, steps_index):
                    state.P[index]['step_scorer_score'] = score

            if self.P_score_type in [None, 'mean']:
                P_score = np.mean([step['step_scorer_score'] for step in state.P])

            elif self.P_score_type in ['min']:
                P_score = np.min([step['step_scorer_score'] for step in state.P])

            elif self.P_score_type in ['tree']:
                sent2score = {}
                for step in state.P: # P in order
                    for p in step['pre_sent']:
                        if state.sent2id[p].startswith('sent'):
                            sent2score[p] = 1.0
                        else:
                            assert p in sent2score
                    sent2score[step['con_sent']] = min(step['step_scorer_score'] , np.min([sent2score[p] for p in step['pre_sent']]))
                
                P_score = np.max([sent2score[con] for con in highest_conclusions])
            
            else:
                raise NotImplementedError

        # ----- score of H -----
        if len(state.P) == 0 or len(highest_conclusions) == 0:
            H_score = 0.0
        else:
            
            # score each intermeidate conclusion
            if self.H_score_type in [None, 'bleurt', 'bleurt+step_scorer']:
                steps_index = []
                bleurt_inputs = []
                for si, step in enumerate(state.P):
                    if 'H_bleurt_score' not in step.keys():
                        steps_index.append(si)
                        bleurt_inputs.append(step['con_sent'])
                        
                if bleurt_inputs:
                    scores = self.bleurt_scorer.score(references = [state.H]*len(bleurt_inputs), 
                                                    candidates = bleurt_inputs)
                    for score, index in zip(scores, steps_index):
                        state.P[index]['H_bleurt_score'] = score
                        
            if self.H_score_type in ['step_scorer', 'bleurt+step_scorer']:
                ## we replace the highest_conclusions with H are verifier the step
                steps_index = []
                step_inputs = []
                for si, step in enumerate(state.P):
                    if 'H_step_scorer_score' not in step.keys():
                        steps_index.append(si)
                        step_inputs.append({
                            'pre_sent': step['pre_sent'],
                            'con_sent': state.H,
                        })
                        
                if step_inputs:
                    scores = self.verify_steps(step_inputs)
                    for score, index in zip(scores, steps_index):
                        state.P[index]['H_step_scorer_score'] = score
      
            # calculate the H score
            if self.H_score_type in [None, 'bleurt']:
                con2score_b = {step['con_sent']:step['H_bleurt_score'] for step in state.P}
                proved_scores = [con2score_b[con] for con in highest_conclusions]
                H_score = np.max(proved_scores)

            elif self.H_score_type in ['step_scorer']:
                con2score_s = {step['con_sent']:step['H_step_scorer_score'] for step in state.P}
                proved_scores = [con2score_s[con] for con in highest_conclusions]
                H_score = np.max(proved_scores)

            elif self.H_score_type in ['bleurt+step_scorer']:
                
                con2score_b = {step['con_sent']:step['H_bleurt_score'] for step in state.P}
                proved_scores = [con2score_b[con] for con in highest_conclusions]
                H_score_b = np.max(proved_scores)

                con2score_s = {step['con_sent']:step['H_step_scorer_score'] for step in state.P}
                proved_scores = [con2score_s[con] for con in highest_conclusions]
                H_score_s = np.max(proved_scores)

                H_score = 0.5 * (H_score_b + H_score_s)

                verifer_info['H_score_b'] = H_score_b
                verifer_info['H_score_s'] = H_score_s


            else:
                raise NotImplementedError


        # ----- score of the whole state -----
        if self.merge_strategy in ['None', None, 'P+H']:
            state_score = 0.5 * (P_score + H_score)

        elif self.merge_strategy in ['P']:
            state_score = P_score
        elif self.merge_strategy in ['H']:
            state_score = H_score
        else:
            raise NotImplementedError

        if return_dict:
            verifer_info.update({
                'state_score':state_score,
                'P_score':P_score,
                'H_score':H_score,
            })
            return verifer_info
        else:
            return state_score

    def verify_steps(self, steps):
        step_scores = self.step_scorer(steps)
        return step_scores

    def verify_get_root_node(self, state):
        highest_conclusions = state.get_pred_highest_conclusions()
        
        root_node = None
        if len(state.P) == 0 or len(highest_conclusions) == 0:
            return root_node
        
        if len(highest_conclusions) == 1:
            root_node = highest_conclusions[0]
            return root_node
        
        
        _ = self.verify_a_state(state)
        
        if self.H_score_type in [None, 'bleurt']:
            con2score_b = {step['con_sent']:step['H_bleurt_score'] for step in state.P}
            proved_scores = [con2score_b[con] for con in highest_conclusions]
            root_node = highest_conclusions[np.argmax(proved_scores)]

        elif self.H_score_type in ['step_scorer']:
            con2score_s = {step['con_sent']:step['H_step_scorer_score'] for step in state.P}
            proved_scores = [con2score_s[con] for con in highest_conclusions]
            root_node = highest_conclusions[np.argmax(proved_scores)]

        elif self.H_score_type in ['bleurt+step_scorer']:

            con2score_b = {step['con_sent']:step['H_bleurt_score'] for step in state.P}
            proved_scores_b = np.array([con2score_b[con] for con in highest_conclusions])

            con2score_s = {step['con_sent']:step['H_step_scorer_score'] for step in state.P}
            proved_scores_s = np.array([con2score_s[con] for con in highest_conclusions])

            root_node = highest_conclusions[np.argmax(proved_scores_b+proved_scores_s)]
        
        else:
            raise NotImplementedError

        return root_node
