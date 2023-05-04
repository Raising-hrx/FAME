import os, sys, copy
import json


def chunk(it, n):
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c
        
# -------- load data util --------
def load_entailmentbank(task='task_1', split='train', version='v3', base_path = None):
    assert task in ['task_1', 'task_2', 'task_3']
    assert split in ['train', 'dev', 'test']
    assert version in ['v2', 'v3', None]
    
    if base_path is None:
        base_path = '../data'


    if version:
        path = os.path.join(base_path, f'entailment_trees_emnlp2021_data_{version}/dataset/{task}/{split}.jsonl')
    else:
        path = os.path.join(base_path, f'dataset/{task}/{split}.jsonl')

    datas = [json.loads(line) for line in open(path).readlines()]
    
    print(f"Loading EntailmentBank data from: {path}")
    return datas


# -----utils for parse annotated data and predicted trees -----
def get_sent_dict(item): return item['meta']['triples'] # return dict {sid:sent}
def get_int_dict(item): return item['meta']['intermediate_conclusions']
def get_hid(item): return item['meta']['hypothesis_id']
def get_id2sent(item):
    id2sent = {}
    id2sent.update(item['meta']['triples'])
    id2sent.update(item['meta']['intermediate_conclusions'])
    id2sent['hypothesis'] = item['hypothesis']
    return id2sent


def parse_proof(proof):
    step_proof = []
    for step in proof.split('; '):
        if not step: continue
        step = step.split(':')[0]
        tmp = [step.split(' -> ')[0].replace(' ','').split('&'),
              step.split(' -> ')[1].replace(' ','')]
        step_proof.append(tmp)
            
    return step_proof

def get_node(idx,node_list):
    """
    find the node with id from the node_list
    """
    node = None
    for item in node_list:
        if item['id'] == idx:
            node = item
    return node

def get_tree(idx,node_list):
    """
    rerank the node_list
    make the idx node as root node
    only retrain nodes in the tree
    """
    childrens = []
    node = get_node(idx,node_list)
    
    if node:
        for child_idx in node['pre']:
            childrens += get_tree(child_idx,node_list)

        return [node] + childrens
    else:
        return None
    
def get_gt_node_list(item):
    """
    load the ground truth tree from the orignal dataset item
    # add full stop
    """
    node_list = []
    for sent_id, sent in get_sent_dict(item).items():
        if sent_id.startswith('sent'):
            node_list.append({
                'id':sent_id,
                # 'sent':add_fullstop(sent),
                'sent':sent,
                'pre':[],
            })

    step_proof = parse_proof(item['meta']['step_proof'])
    for sent_id, sent in get_int_dict(item).items(): 
        if sent_id == item['meta']['hypothesis_id']:
            index = [step[1] for step in step_proof].index('hypothesis')
        else:
            index = [step[1] for step in step_proof].index(sent_id)
        node_list.append({
            'id':sent_id,
            # 'sent':add_fullstop(sent),
            'sent':sent,
            'pre':step_proof[index][0],
        })   
        
    node_list = get_tree(item['meta']['hypothesis_id'],node_list)
    
    node_list[0]['id'] = 'hypothesis'
        
    return node_list

def get_leaves_ids(idx,node_list):
    """
    get all leaf nodes in the tree (sentX / pre in none)
    """
    leaves_ids = []
    node = get_node(idx,node_list)
    
    if node:
        if node['id'].startswith('sent') or len(node['pre']) == 0:
            return [node['id']]
        else:
            for child_idx in node['pre']:
                leaves_ids += get_leaves_ids(child_idx,node_list)
                
        return leaves_ids
    else:
        return []

def get_all_leaves_ids(node_list):
    return [node['id'] for node in node_list if node['id'].startswith('sent')]

def print_node_tree(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        print(node['id']+': '+node['sent'])
    else:
        print('\t| '*(depth-1)+'\t|- '+node['id'] +': '+node['sent'])
    for child_idx in node['pre']:
        print_node_tree(child_idx,node_list,depth+1)
        
def print_node_tree_with_type(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        print(f"{node['id']}: {node['sent']} ({node.get('step_type','None')}  pred: {node.get('orig_sent','')}) ")
    else:
        print('\t| '*(depth-1)+'\t|- '+ f"{node['id']}: {node['sent']} ({node.get('step_type','None')}) ")
    for child_idx in node['pre']:
        print_node_tree_with_type(child_idx,node_list,depth+1)
        

def print_node_tree_str(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        s = f"{node['id']}: {node['sent']}"
    else:
        s = '\t| '*(depth-1)+'\t|- '+node['id'] +': '+node['sent']
    for child_idx in node['pre']:
        s_child = print_node_tree_str(child_idx,node_list,depth+1)
        s = s + '\n' + s_child
        
    return s
 
def linearize_node_list(node_list):
    proof = ''
    for node in node_list[::-1]:
        if node['id'].startswith('sent'): continue
        proof += f"{' & '.join(node['pre'])} -> {node['id']}"
        if node['id'] != 'hypothesis':
            proof += ': '+ remove_fullstop(node['sent']) +'; '
        else:
            proof += '; '
    return proof

def convert_to_result(tree, data_item, make_new_int = False):
    # tree -> csv result
    # tree: predict node list
    # data_item: gold node list
    if tree is None:
        result = {
            'id' : data_item['id'],
            'steps' : [],
            'texts' : {},
            'proof' : '',
        }
        return result, {}
    
    tree = copy.deepcopy(tree)
    
    if make_new_int:
        # make new intermediate sentences id
        new_int_id_dict = {}
        new_counter = 0
        for node in tree:
            if node['id'].startswith('sent'):
                new_int_id_dict[node['id']] = node['id']
            else:
                new_int_id_dict[node['id']] = f"int{new_counter}"
                new_counter += 1 

        for node in tree:
            node['id'] = new_int_id_dict[node['id']]
            node['pre'] = [new_int_id_dict[idx] for idx in node['pre']]
        
    # convert tree to result item
    result = {}
    result['id'] = data_item['id']
    result['steps'] = []
    result['texts'] = {}
    
    root = tree[0]
    result['texts']['hypothesis'] = data_item['hypothesis']
    result['steps'].append([root['pre'],'hypothesis'])
    
    for node in tree[1:]:
        if node['id'].startswith('sent'):
            result['texts'][node['id']] = node['sent']
        else:
            result['texts'][node['id']] = node['sent']
            result['steps'].append([node['pre'],node['id']])
        
    proof = ''
    for step_pre, step_con in result['steps'][::-1]:
        proof += " & ".join(step_pre) + " -> " + step_con
        if step_con != 'hypothesis':
            proof += ': '+ remove_fullstop(result['texts'][step_con]) +'; '
        else:
            proof += '; '
            
    result['proof'] = proof

    return result

def rename_node(tree, id_map = None):
    # rename the node id of the tree given id_map
    # if id_map is None, id_map{'x'} = 'rename_x'
    if id_map is None:
        id_map = {}
        for node in tree:
            nid = node['id']
            new_nid = 'rename_' + nid
            id_map[nid] = new_nid
    else:
        for node in tree:
            nid = node['id']
            if nid not in id_map.keys():
                id_map[nid] = nid
    
    new_tree = copy.deepcopy(tree)
    for node in new_tree:
        if node['id'] in id_map:
            node['id'] = id_map[node['id']]
        node['pre'] = [id_map[nid] if nid in id_map else nid for nid in node['pre']]
    return new_tree

### -------- nx_graph utils --------
import networkx as nx
from IPython.display import display, Image
import textwrap

def visualize_nx_graph(nx_graph, text_width = 40):
    graph = nx.nx_pydot.to_pydot(nx_graph)
    graph.set_rankdir('BT')
    
    for node in graph.get_nodes():
        # shape & color
        if node.get('id').startswith('sent'):
            node.set_shape("box")
            node.set_color("lightsalmon2")
            node.set_style("filled")
        else:
            node.set_shape("box")
            node.set_color("lightblue")
            node.set_style("filled")
        
        # text
        label = f"{node.get('id')}: {node.get('sent')}"
        label = textwrap.fill(label,width=text_width)
        node.set_label(label)
    
    if 'ipykernel' in sys.modules:
        display(Image(graph.create_png()))
    else:
        return graph.create_png()
    
    
    
def node_list_to_nx_graph(node_list):
    """
    convert the node-list tree to nx-graph tree
    """
    id2sent = {node['id']:node['sent'] for node in node_list}

    G = nx.DiGraph()
    G.add_nodes_from([(node['sent'], {"id":node['id'], 'sent':node['sent']}) for node in node_list])
    for node in node_list:
        for p in node['pre']:
            G.add_edge(id2sent[p], node['sent'])

    return G

def get_gt_nx_graph(data_item):
    return node_list_to_nx_graph(get_gt_node_list(data_item))

def nx_graph_to_node_list(nx_graph, root_node = None):

    if len(nx_graph) == 0:
        return []

    # Find the root of tree
    if root_node not in nx_graph:
        root_node = None

    if root_node is None:
        # find the last node of the longest_path in nx_graph
        longest_path = nx.dag_longest_path(nx_graph)
        root_node = longest_path[-1]
        # # find one of the node which has in_edge but does not have out_edge
        # for node in nx_graph:
        #     if nx_graph.in_degree(node) > 0 and nx_graph.out_degree(node) == 0:
        #         root_node = str(node)
        #         break
    # assert root_node is not None, "Can not find root node in the graph"
    # assert root_node in nx_graph, f"Root node in the graph; root node: {root_node}"

    assert root_node is not None

    
    # convert to tree in node_list format
    root_id = nx_graph.nodes[root_node]['id']
    
    node_list = []

    for node in nx_graph:
        new_node = {
            'id': nx_graph.nodes[node]['id'],
            'sent':node,
            'pre': [nx_graph.nodes[pre_node]['id'] for pre_node in list(nx_graph.predecessors(node))],
            'pre_sent': [nx_graph.nodes[pre_node]['sent'] for pre_node in list(nx_graph.predecessors(node))],

        }
        node_list.append(new_node)

    node_list_tree = get_tree(root_id, node_list)
    
    return node_list_tree

        
# -------- Sentence process utils --------
def add_fullstop(sent):
    if sent.endswith('.'):
        return sent
    else:
        return sent+'.'

def remove_fullstop(sent):
    if sent.endswith('.'):
        return sent[:-1]
    else:
        return sent

def decapitalize(sent):
    return sent[0].lower() + sent[1:]


def LCstring(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                result = max(result,res[i][j])  
    return result


def sent_overlap(sent1,sent2,spacy_nlp,thre=-1):
    
    spacy_nlp.Defaults.stop_words -= {"using", "show","become","make","down","made","across","put","see","move","part","used"}
    
    doc1 = spacy_nlp(sent1)
    doc2 = spacy_nlp(sent2)    
    
    word_set1 = set([token.lemma_ for token in doc1 if not (token.is_stop or token.is_punct)])
    word_set2 = set([token.lemma_ for token in doc2 if not (token.is_stop or token.is_punct)])
    
    if thre == -1:
        # do not use LCstring
        if len(word_set1.intersection(word_set2)) > 0:
            return True
        else:
            return False

    # use LCstring
    max_socre = -1
    for word1 in word_set1:
        for word2 in word_set2:
            lcs = LCstring(word1,word2)
            score = lcs / (min(len(word1),len(word2))+1e-10)
            max_socre = score if score > max_socre else max_socre
            
    if max_socre > thre:
        return True
    else:
        return False

import json, string, re, os

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def same_sent(sent1, sent2):
    return normalize_answer(sent1) == normalize_answer(sent2)

def sent_IoU(sent1,sent2,spacy_nlp):
    sent1 = normalize_answer(sent1)
    sent2 = normalize_answer(sent2)
    
    spacy_nlp.Defaults.stop_words -= {"using", "show","become","make","down","made","across","put","see","move","part","used"}
    
    doc1 = spacy_nlp(sent1)
    doc2 = spacy_nlp(sent2)    
    
    word_set1 = set([token.lemma_ for token in doc1 if not (token.is_stop or token.is_punct)])
    word_set2 = set([token.lemma_ for token in doc2 if not (token.is_stop or token.is_punct)])
    
    inter = 0
    for word1 in word_set1:
        for word2 in word_set2:
            lcs = LCstring(word1,word2)
            if lcs / min(len(word1),len(word2)) > 0.6:
                inter += 1
                break
    # print(word_set1)
    # print(word_set2)
    
    iou = inter / (len(word_set1.union(word_set2))+1e-10)
    return iou

# -----Evaluation utils -----
def Jaccard(set1,set2):
    set1 = set(set1)
    set2 = set(set2)
    Intersection = len(set1.intersection(set2))
    Union = len(set1.union(set2))
    
    return Intersection / (Union + 1e-20)

def div(num, denom):
    return num / denom if denom > 0 else 0

def compute_f1(matched, predicted, gold):
    # 0/0=1; x/0=0
    precision = div(matched, predicted)
    recall = div(matched, gold)
    f1 = div(2 * precision * recall, precision + recall)
    
    if predicted == gold == 0:
        precision = recall = f1 = 1.0
    
    return precision, recall, f1



