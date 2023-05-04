import os
import sys
import random
import copy
import os.path as osp
import json
from collections import defaultdict
import argparse

import numpy as np
import torch
import transformers

from torch.utils.data import DataLoader

import sentence_transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers import losses, models, util
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

import tqdm

from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from tree_utils import *
from Retriever import Dense_Retriever
from retrieval_metric import evaluate_retrieval


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


class AutoModelForSentenceEmbedding(nn.Module):
    # ref: https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/train_script.py
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)

        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def save_as_sentence_transformers(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)

        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        # save addition file for the sentence_transformers to load model directly
        with open(osp.join(output_path, 'sentence_bert_config.json'), 'w') as f:
            json.dump({"max_seq_length": 384,"do_lower_case": False,}, f)

        with open(osp.join(output_path, 'modules.json'), 'w') as f:
            json.dump([{
                            "idx": 0,
                            "name": "0",
                            "path": "",
                            "type": "sentence_transformers.models.Transformer"
                        },
                        {
                            "idx": 1,
                            "name": "1",
                            "path": "1_Pooling",
                            "type": "sentence_transformers.models.Pooling"
                        },
                        {
                            "idx": 2,
                            "name": "2",
                            "path": "2_Normalize",
                            "type": "sentence_transformers.models.Normalize"
                        }
                        ], f)

        with open(osp.join(output_path, 'config_sentence_transformers.json'), 'w') as f:
            json.dump({
                "__version__": {
                            "sentence_transformers": sentence_transformers.__version__,
                            "transformers": transformers.__version__,
                            "pytorch": torch.__version__
                        }
                        }, f)

        os.makedirs(osp.join(output_path, '1_Pooling'), exist_ok=True)
        os.makedirs(osp.join(output_path, '2_Normalize'), exist_ok=True)
        with open(osp.join(output_path, '1_Pooling', 'config.json'), 'w') as f:
            json.dump({
                        "word_embedding_dimension": 768,
                        "pooling_mode_cls_token": False,
                        "pooling_mode_mean_tokens": True,
                        "pooling_mode_max_tokens": False,
                        "pooling_mode_mean_sqrt_len_tokens": False
                        }, f)


      
def make_retriever_training_samples_contrastive(datas, random_neg_pos_ratio = 2, hard_neg_pos_ratio = 2,
                                                retriever = None, query_candidate_type = None):
    """
    random_neg_pos_ratio: len(random_nagtive_samples) / len(positive_samples)
    hard_neg_pos_ratio: len(hard_nagtive_samples) / len(positive_samples)
    
    retriever: model for collecting hard negative
    """
    all_samples_by_item = []
    
    for data_item in datas:
        # -------- collect item information --------
        H = data_item['hypothesis']
        I = list(data_item['meta']['intermediate_conclusions'].values())
        I.remove(H)
        S = list(data_item['meta']['triples'].values())
        gold_tree_text = [H] + I + S
        
        candidate_queries = [H]
        
        if query_candidate_type == "HI":
            candidate_queries = [H] + I
        if query_candidate_type == "HIS":
            candidate_queries = gold_tree_text

        id2sent = copy.deepcopy(data_item['meta']['triples'])
        id2sent.update(data_item['meta']['intermediate_conclusions'])

        same_step_text = defaultdict(list)
        for node in get_gt_node_list(data_item):
            if len(node['pre']) == 0: continue
            con = node['sent']
            pre = [id2sent[p] for p in node ['pre']]

            same_step_text[con] += pre
            for p in pre:
                same_step_text[p] += [con] + [other_p for other_p in pre if other_p != p]
                 
        # -------- positive samples --------
        positive_samples = []
        for query in candidate_queries:
            for target in S:
                if query == target: continue
                if target in same_step_text[query]:
                    positive_samples.append(InputExample(texts=[query, target], label = 1.0)) # same_step_score
                else:
                    positive_samples.append(InputExample(texts=[query, target], label = 1.0)) # same_tree_score
                    
        # ------- negative samples --------
        # part1: random sample from corpus
        random_nagative_samples = []
        if random_neg_pos_ratio > 0:
            for query in candidate_queries:
                negs = random.sample(list(corpus.values()), int(len(positive_samples) * random_neg_pos_ratio))
                for neg_target in negs:
                    if neg_target not in gold_tree_text:
                        random_nagative_samples.append(InputExample(texts=[query, neg_target], label = 0.0))
            if len(random_nagative_samples) > len(positive_samples) * random_neg_pos_ratio:
                random_nagative_samples = random.sample(random_nagative_samples, int(len(positive_samples) * random_neg_pos_ratio))

        # part2: hard negative from model which has not been fine-tuned
        hard_nagative_samples = []
        hard_negative_topk = 2 * int(len(positive_samples) * hard_neg_pos_ratio)
        if hard_neg_pos_ratio > 0:
            for query in candidate_queries:
                retrieval_result = retriever.search(query, hard_negative_topk)
                negs = [r['text'] for r in retrieval_result]
                for neg_target in negs:
                    if neg_target not in gold_tree_text:
                        hard_nagative_samples.append(InputExample(texts=[query, neg_target], label = 0.0))
            if len(hard_nagative_samples) > len(positive_samples) * hard_neg_pos_ratio:
                hard_nagative_samples = random.sample(hard_nagative_samples, int(len(positive_samples) * hard_neg_pos_ratio))
                    
        negative_samples = random_nagative_samples + hard_nagative_samples
        
        # ------- collect all samples --------
        all_samples_by_item.append({
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
        })
    
    return all_samples_by_item

def queue_get(samples_by_item, batch_size, triple=False):
    # return a bacth for training
    # triple = True (anchor, positive, negative); triple = False (anchor, positive)
    
    batch = []
    
    if triple == False: # (anchor, positive)
        for sample in random.sample(samples_by_item, batch_size):
            query, target = random.choice(sample['positive_samples']).texts
            batch.append([query, target])

    else: # (anchor, positive, negative)
        for sample in random.sample(samples_by_item, batch_size):
            query, pos_target = random.choice(sample['positive_samples']).texts
            _, neg_target = random.choice(sample['negative_samples']).texts
            
            batch.append([query, pos_target, neg_target]) 
        
    return batch
        



def train_function(model, tokenizer, args, queue_get, samples_by_item):
    # ref: https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/train_script.py
    
    # steps num_warmup_steps lr 
    # scale save_steps output
    device = 'cuda'
    
    ### Train Loop
    model = model.to(device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=True)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, # 100,
        num_training_steps=args.steps,
    )
    
    # Now we train the model
    model.train()

    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1
   
    dev_best_metric = -100
    test_best_metric = -100


    # load initial model and eval
    bi_encoder = SentenceTransformer(args.model_name_or_path)
    dr = Dense_Retriever(corpus,bi_encoder,buffer_file=None)
    dev_task1 = load_entailmentbank('task_1','dev')
    dev_eval_result = eval(dev_task1, dr)
    test_task1 = load_entailmentbank('task_1','test')
    test_eval_result = eval(test_task1, dr)
    with open(args.metric_file, 'a') as f:
        f.write(json.dumps(
            {'step':0, 
            'dev_eval_result': dev_eval_result, 
            'test_eval_result': test_eval_result,}
        ) + '\n')

    for global_step in tqdm.trange(args.steps):
        #### Get the batch data
        # batch = queue.get()
        # print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))
        batch = queue_get(samples_by_item, args.bs, args.triple)

        if len(batch[0]) == 2: #(anchor, positive)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            ### Compute embeddings
            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))

            ### Compute similarity scores 512 x 512
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## Symmetric loss as in CLIP
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2

        else:   #(anchor, positive, negative)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text3 = tokenizer([b[2] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            embeddings_a  = model(**text1.to(device))
            embeddings_b1 = model(**text2.to(device))
            embeddings_b2 = model(**text3.to(device))

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

            ### Compute similarity scores 512 x 1024
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## One-way loss
            loss = cross_entropy_loss(scores, labels)

        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()


        # eval and save model
        if (global_step+1) % args.save_steps == 0:

            # save model
            output_path = os.path.join(args.exp_dir, "latest") # TODO
            print("save model: "+output_path)
            model.save_as_sentence_transformers(output_path)

            # load model and eval
            bi_encoder = SentenceTransformer(output_path)
            dr = Dense_Retriever(corpus,bi_encoder,buffer_file=None)

            dev_task1 = load_entailmentbank('task_1','dev')
            dev_eval_result = eval(dev_task1, dr)
            dev_new_metric = dev_eval_result['R@25'] + 0.1*dev_eval_result['AllCorrect@25']
            if dev_new_metric > dev_best_metric:
                dev_best_metric = dev_new_metric
                model.save_as_sentence_transformers(os.path.join(args.exp_dir, "best_model"))

            test_task1 = load_entailmentbank('task_1','test')
            test_eval_result = eval(test_task1, dr)
            test_new_metric = test_eval_result['R@25'] + 0.1*test_eval_result['AllCorrect@25']
            if test_new_metric > test_best_metric:
                test_best_metric = test_new_metric
                model.save_as_sentence_transformers(os.path.join(args.exp_dir, "_best_model"))


            with open(args.metric_file, 'a') as f:
                f.write(json.dumps(
                    {'step':global_step+1, 
                    'dev_eval_result': dev_eval_result, 
                    'test_eval_result': test_eval_result,}
                ) + '\n')


def eval(data_task1, retriever):
    golds = {}
    for date_idx, data_item in enumerate(data_task1):
        qid = data_item['id']
        gold_retrieval = {}
        for sent_id, sent in data_item['meta']['triples'].items():
            gold_retrieval[sent] = 1
        golds[qid+str(date_idx)] = gold_retrieval

    preds = {}
    for date_idx, data_item in enumerate(data_task1):
        qid = data_item['id']
        Q = data_item['question']
        A = data_item['answer']    
        H = data_item['hypothesis']  
        
        query = H
        retrieval_result = retriever(query, n=25)
        
        pred_retrieval = {
            i['text']:i['score']
            for i in retrieval_result
        }
        
        preds[qid+str(date_idx)] = pred_retrieval
        
    eval_result = evaluate_retrieval(preds, golds)
    return eval_result




def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training')

    # dateset
    parser.add_argument('--random_neg_pos_ratio', type=float, default=0, help='')
    parser.add_argument('--hard_neg_pos_ratio', type=float, default=0, help='')
    parser.add_argument('--query_candidate_type', type=str, default=None, help='')

    # model
    parser.add_argument("--model_name_or_path", type=str, default="sentence-transformers/all-mpnet-base-v2", help="")  
    parser.add_argument('--max_length', type=int, default=256, )

    # optimization
    parser.add_argument('--bs', type=int, default=16, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--steps', type=int, default=1, help='')
    parser.add_argument('--scale', type=float, default=1.0, help='')
    parser.add_argument('--num_warmup_steps', type=int, default=100, help='')

    parser.add_argument('--triple', type=str, default='False')
    parser.add_argument('--save_steps', type=int, default=100)

    # seed
    parser.add_argument('--seed', type=int, default=3407, help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='./exp')
    parser.add_argument("--code_dir", type=str, default='./code')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args

### ---------- main ----------
args = get_params()
if args.seed == 0:
    args.seed = random.randint(1,1e4)

assert args.triple in ['False', 'True']
args.triple = False if args.triple=='False' else True

args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())
os.makedirs(args.exp_dir, exist_ok=True)

# make metrics.json for logging metrics
args.metric_file = osp.join(args.exp_dir, 'metrics.json')
open(args.metric_file, 'a').close()

# dump config.json
with open(osp.join(args.exp_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

# backup scripts
os.system(f'cp -r {args.code_dir} {args.exp_dir}')


# set random seed before init model
torch.backends.cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if torch.cuda.device_count() > 1:
#     torch.cuda.manual_seed_all(args.seed)


# load corpus
path = '../data/entailment_trees_emnlp2021_data_v3/supporting_data/preprocessed_corpus.json'
corpus = json.load(open(path))

hard_neg_retriever = None
if args.hard_neg_pos_ratio > 0:
    bi_encoder = SentenceTransformer(args.model_name_or_path)
    hard_neg_retriever = Dense_Retriever(corpus,bi_encoder,buffer_file=None)

train_datas = load_entailmentbank('task_1','train')
train_samples_by_item = make_retriever_training_samples_contrastive(train_datas, 
                                                                   random_neg_pos_ratio = args.random_neg_pos_ratio, 
                                                                   hard_neg_pos_ratio = args.hard_neg_pos_ratio,
                                                                   retriever = hard_neg_retriever,
                                                                   query_candidate_type = args.query_candidate_type)


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForSentenceEmbedding(args.model_name_or_path, tokenizer)
train_function(model, tokenizer, args, 
                queue_get=queue_get, samples_by_item=train_samples_by_item)