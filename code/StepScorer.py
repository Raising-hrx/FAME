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

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from transformers.optimization import Adafactor,AdamW,get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score

from tree_utils import *
from exp_utils import create_optimizer, create_scheduler

##### hrx experiment utils
import socket
import getpass
def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname, file_only=False):
    if file_only:
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
#####

class StepDataset(Dataset):
    
    def __init__(self, data_file, input_form = None):
        datas = [json.loads(line) for line in open(data_file).readlines()]

        datas =  self.data_preprocess(datas)
            
        self.datas = datas
            
        
    def __getitem__(self, index):
        return self.datas[index]
    
    def __len__(self):
        return len(self.datas)

    @classmethod
    def data_preprocess(cls, datas):
        for item in datas:
            premises = [add_fullstop(p).lower() for p in item['pre_sent']]
            premises = " ".join(premises)
            conclusion = add_fullstop(item['con_sent']).lower()
            item['src'] = f"premises: {premises} conclusion: {conclusion}"
            # item['src'] = f"{premises} </s> </s> {conclusion}"
        return datas
        




def train_one_step(batch, model, tokenizer, args):

    model.train()
    
    # process batch data
    labels = [item['label'] for item in batch]
    input_sents = [item['src'] for item in batch]

    input_batch = tokenizer(
            input_sents,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            max_length=args.max_length,
            truncation=True,)

    if args.cls_problem_type == 'single_label_classification':
        # labels.dtype=torch.long; defalut:single_label_classification with softmax if not set model.config.problem_type
        input_batch['labels'] = torch.tensor(labels)
    else:
        raise NotImplementedError

    input_batch = input_batch.to(model.device)
    
    # forward
    model_return = model(**input_batch)

    return model_return['loss']

def eval_model(model,data_loader, tokenizer):
    
    model.eval()
    # torch.cuda.empty_cache()
    
    gold_labels = []
    pred_labels = []
    
    for batch in data_loader:
        
        # process batch data
        labels = [item['label'] for item in batch]
        input_sents = [item['src'] for item in batch]

        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=args.max_length,
                truncation=True,)
        
        input_batch = input_batch.to(model.device)

        # forward
        model_return = model(**input_batch)
        
        logits = model_return['logits']
        preds = torch.argmax(logits,dim=1).detach().cpu().numpy().tolist()
        
        gold_labels += labels
        pred_labels += preds
        
    # eval
    acc = np.sum(np.array(gold_labels) == np.array(pred_labels)) / len(gold_labels)

    micro_acc = precision_score(y_true=gold_labels,y_pred=pred_labels,average='micro')
    macro_acc = precision_score(y_true=gold_labels,y_pred=pred_labels,average='macro')

    assert acc == micro_acc

    return {
        'micro_acc': micro_acc, 
        'macro_acc': macro_acc,
    }

def load_step_scorer(exp_dir, model_name = 'best_model.pth'):
    print(f"Loading model from {osp.join(exp_dir, model_name)}")
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)

    # load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model.config.update(model_config)

    # load trained parameters
    state_dict = torch.load(osp.join(exp_dir, model_name), map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model, tokenizer, args


def predict_step_scorer(model, tokenizer, datas, bs = 4):
    """
    datas: List of {'pre_sent': ..., 'con_sent'}
    """
    model.eval()
    # torch.cuda.empty_cache()
    
    inputs = []
    pred_labels = []
    pred_scores = []
    
    datas = StepDataset.data_preprocess(datas)

    for batch in chunk(datas, bs):
        
        # process batch data
        input_sents = [item['src'] for item in batch]

        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=256,
                truncation=True,)
        
        input_batch = input_batch.to(model.device)

        # forward
        model_return = model(**input_batch)
        
        logits = model_return['logits']
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        preds_s= logits.softmax(dim=1).max(dim=1).values.detach().cpu().numpy().tolist()

        inputs += input_sents
        pred_labels += preds
        pred_scores += preds_s

    pred_scores = [
        s if la == 1 else 1.0 - s
        for la, s in zip(pred_labels, pred_scores)
    ]

    return pred_scores

class StepScorer():
    def __init__(self, exp_dir, model_name = 'best_model.pth', device='cuda'):

        # load model
        model, tokenizer, args = self.load_step_scorer(exp_dir, model_name)
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device


    def load_step_scorer(self, exp_dir, model_name = 'best_model.pth'):
        print(f"Loading model from {osp.join(exp_dir, model_name)}")
        # read config
        config = json.load(open(osp.join(exp_dir,'config.json')))
        model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
        args = argparse.Namespace(**config)

        # load model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        model.config.update(model_config)

        # load trained parameters
        state_dict = torch.load(osp.join(exp_dir, model_name), map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model, tokenizer, args

    def predict_step_scorer(self, datas, bs = 4):
        """
        datas: List of {'pre_sent': ..., 'con_sent'}
        """
        model, tokenizer = self.model, self.tokenizer

        model.eval()
        
        inputs = []
        pred_labels = []
        pred_scores = []
        
        datas = StepDataset.data_preprocess(datas)

        for batch in chunk(datas, bs):
            
            # process batch data
            input_sents = [item['src'] for item in batch]

            input_batch = tokenizer(
                    input_sents,
                    add_special_tokens=True,
                    return_tensors='pt',
                    padding='longest',
                    max_length=256,
                    truncation=True,)
            
            input_batch = input_batch.to(model.device)

            # forward
            model_return = model(**input_batch)
            
            logits = model_return['logits']
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_s= logits.softmax(dim=1).max(dim=1).values.detach().cpu().numpy().tolist()

            inputs += input_sents
            pred_labels += preds
            pred_scores += preds_s

        pred_scores = [
            s if la == 1 else 1.0 - s
            for la, s in zip(pred_labels, pred_scores)
        ]

        return pred_scores


    def __call__(self, *args, **kwargs):
        return self.predict_step_scorer(*args, **kwargs)


def run(args):

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    log.info(f"classify problem type:{args.cls_problem_type} (single for softmax+nllloss; multi for sigmoid+bceloss)")

    log.info("loading data")
    train_dataset = StepDataset(args.train_data)
    dev_dataset = StepDataset(args.dev_data)

    log.info(f"Length of training dataest: {len(train_dataset)}")
    log.info(f"Length of dev dataest: {len(dev_dataset)}")

        
    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.bs,
                            shuffle = True,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    dev_loader = DataLoader(dataset = dev_dataset,
                            batch_size = args.bs,
                            shuffle = False,
                            num_workers = 4,
                            collate_fn = lambda batch: batch)
    
    args.num_training_steps = args.epochs * len(train_loader)
    args.eval_iter  = round(args.eval_epoch * len(train_loader))
    args.report_iter  = round(args.report_epoch * len(train_loader))
    
    log.info(f"number of iteration / epoch : {len(train_loader)}")

    log.info("loading model")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels = args.num_labels, ignore_mismatched_sizes=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels = args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    
    model.config.update({'problem_type': args.cls_problem_type,})
    
    model = model.to(device)

    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)

    log.info("start training")
    global_iter = 0
    loss_list = []
    best_metric = -100

    for epoch_i in range(args.epochs):
        
        for batch in train_loader:
            loss = train_one_step(batch,model,tokenizer,args)
            
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            
            global_iter += 1
            
            
            if not global_iter % args.eval_iter:
                eval_result = eval_model(model,dev_loader, tokenizer)
                micro_acc, macro_acc = eval_result['micro_acc'], eval_result['macro_acc']

                new_metric = micro_acc

                log.info(f"Iteration {global_iter} test micro_acc:{micro_acc:.4f} macro_acc:{macro_acc:.4f}")

                if best_metric < new_metric:
                    best_metric = new_metric
                    log.info(f"-----Iteration {global_iter} get best metric {best_metric:.4f}-----")

                    # save model
                    if args.save_model:
                        save_path = osp.join(args.exp_dir,'best_model.pth')
                        torch.save(model.state_dict(), save_path)
                        log.info(f"Iteration {global_iter} save best model")

                with open(args.metric_file, 'a') as f:
                    f.write(json.dumps(
                        {'iter':global_iter, 'dev_eval_result': eval_result}
                    ) + '\n')

            if not global_iter % args.report_iter:
                log.info(f"Iteration {global_iter} training loss {np.mean(loss_list):.4f}")
                loss_list = []
            else:
                loss_list.append(float(loss.cpu().data))
                
        log.info(f"epoch {epoch_i} finished")


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training')

    # dateset
    parser.add_argument("--train_data", type=str, default='', help="training data file")
    parser.add_argument("--dev_data", type=str, default='', help="dev data file")

    # model
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large", help="")  
    parser.add_argument('--max_length', type=int, default=128, )
    parser.add_argument('--num_labels', type=int, default=2, )
    parser.add_argument("--cls_problem_type", type=str, 
                        default="single_label_classification", 
                        help="single_label_classification for softmax+nllloss, multi_label_classification for sigmoid+BCEloss") 

    # optimization
    parser.add_argument('--bs', type=int, default=16, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
                        
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='./exp')
    parser.add_argument("--code_dir", type=str, default='./code')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--report_epoch', type=float, default=1.0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1,1e4)

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())

    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=True)
    
    # make metrics.json for logging metrics
    args.metric_file = osp.join(args.exp_dir, 'metrics.json')
    open(args.metric_file, 'a').close()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    os.system(f'cp -r {args.code_dir} {args.exp_dir}')

    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}, PID: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd(), os.getpid()))
    log.info('Python info: {}'.format(os.popen('which python').read().strip()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    run(args)

    # make 'done' file
    open(osp.join(args.exp_dir, 'done'), 'a').close()