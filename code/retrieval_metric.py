import ir_measures # https://ir-measur.es/en/latest/measures.html

def evaluate_retrieval(preds, golds):
    """
    golds = {
        'query0': {"doc0": 0, "doc1": 1},
        "query1": {"doc0": 0, "doc1": 2} 
    }
    preds = {
        'query0': {"doc0": 1.2, "doc1": 1.0},
        "query1": {"doc0": 2.4, "doc1": 3.6} 
    }
    """

    result = ir_measures.calc_aggregate([ir_measures.R@5, ir_measures.R@10, ir_measures.R@25, ir_measures.nDCG@25], golds, preds)
    
    # AllCorrect@25
    AC25s = []
    for qid in golds.keys():
        if qid not in preds:
            print(f"evaluate_retrieval missing prediction:{qid}")
        
        R25 = ir_measures.calc_aggregate([ir_measures.R@25], {qid: golds[qid]}, {qid: preds.get(qid, {})})
        R25 = list(R25.values())[0]
        AC25s.append(int(R25 == 1))
    result['AllCorrect@25'] = sum(AC25s) / (len(AC25s)+1e-10)
    
    result = {str(k):v for k,v in result.items()}
    
    return result