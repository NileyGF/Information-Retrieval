import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import structures as st
from Vector_model import Vector_model
from LSI_model import LSI_model
from Boolean_model import Boolean_model


def qrel_by_model(model, collection):
    if model == 'vector'    : m = Vector_model(collection)
    elif model == 'LSI'     : m = LSI_model(collection)
    elif model == 'boolean' : m = Boolean_model(collection)
    else : 
        print("Wrong Model")
        return
    m.collection.load_docs()
    m.collection.load_queries()

    model_rel = {k: [] for k in m.collection.queries_dict.keys()}
    
    for q in model_rel:
        docs_rel = m.query(m.collection.queries_dict[q].query)
        model_rel[q] = [d[0] for d in docs_rel]
    
    return model_rel

# ret = {}
# m = ['vector', 'boolean', 'LSI']
# c = ['cranfield', 'vaswani', 'vaswani_r', 'nfcorpus', 'nfcorpus_r']
# rel_path = os.path.join(os.path.dirname(__file__),'data')
# qrel_by_model(m[1],c[0])
# ret[(m[0],c[0])] = qrel_by_model(m[0],c[0])     # vector - cranfield 
# ret[(m[0],c[1])] = qrel_by_model(m[0],c[1])     # vector - vaswani
# ret[(m[0],c[3])] = qrel_by_model(m[0],c[3])     # vector - nfcorpus
# f = open( os.path.join(rel_path, 'relevance_vector.bin','wb') )
# pickle.dump(ret,f)
# f.close()
# ret = {}
# ret[(m[1],c[0])] = qrel_by_model(m[1],c[0])     # boolean - cranfield
# ret[(m[1],c[1])] = qrel_by_model(m[1],c[1])     # boolean - vaswani
# ret[(m[1],c[3])] = qrel_by_model(m[1],c[3])     # boolean - nfcorpus
# f = open( os.path.join(rel_path, 'relevance_boolean.bin','wb') )
# pickle.dump(ret,f)
# f.close()
# ret = {}
# ret[(m[2],c[0])] = qrel_by_model(m[2],c[0])     # LSI - cranfield
# ret[(m[2],c[2])] = qrel_by_model(m[2],c[2])     # LSI - vaswani_r
# ret[(m[2],c[4])] = qrel_by_model(m[2],c[4])     # LSI - nfcorpus_r
# f = open( os.path.join(rel_path, 'relevance_LSI.bin','wb') )
# pickle.dump(ret,f)
# f.close()

def hits(qrels: list, retrieved: list):
    """ Number of retrieved relevant documents. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
    """
    hits = 0
    for i in range(len(retrieved)):
        for k in range(len(qrels)):
            if retrieved[i] == qrels[k]:
                hits+=1
                break       #it won't match another one
    return hits
def hit_rate(qrels: list, retrieved: list):
    """ Fraction of queries for which at least one relevant document is retrieved. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
    """
    for i in range(len(retrieved)):
        for k in range(len(qrels)):
            if retrieved[i] == qrels[k]:
                return 1
    return 0
def precision(qrels: list, retrieved: list):
    """ Proportion of the retrieved documents that are relevant. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
        
        Precision = r/n
        r: number of retrieved relevant documents
        n: number of retrieved documents
    """
    if len(retrieved) == 0: return 0
    return hits(qrels, retrieved) / len(retrieved)
def recall(qrels: list, retrieved: list):
    """ Ratio between the retrieved documents that are relevant and the total number of relevant documents. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
        Recall = r/R
        r: number of retrieved relevant documents
        R: total number of relevant documents
    """
    if len(qrels) == 0: return 0
    return hits(qrels, retrieved) / len(qrels)
def f_metric(qrels: list, retrieved: list, beta = 1):
    """ Weighted harmonic mean of Precision and Recall. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
        F = ((1+beta^2) * P * R ) / ( beta^2 * P + R )
        beta: weight
        P: precision
        R: recall
    """    
    if beta <= 0: return None
    precision_s = precision(qrels, retrieved)
    recall_s = recall(qrels, retrieved)
    if precision_s == 0 or recall_s == 0: return 0
    return ((1 + beta ** 2) * precision_s * recall_s) / ((beta ** 2) * precision_s + recall_s)
def precision_ranked(qrels: list, retrieved: list):
    """ Proportion of the retrieved documents that are relevant. 
        qrels:       list of relevant documents
        retrieved:   list of retrieved documents
        R-Precision = r/R
        r: number of relevant documents among the top-R retrieved
        R: total number of relevant documents
        R-Precision is equal to recall at the R-th position
    """
    R = min(len(qrels), len(retrieved))
    hits_on_top_R = hits(qrels, retrieved[:R])
    if len(qrels) == 0: return 0
    return hits_on_top_R / len(qrels)
def fallout(qrels: list, retrieved: list, total_docs: int):
    """ Proportion of non-relevant documents retrieved,
        out of all non-relevant documents available 
        fallout = nr/nn
        nr: number of non-relevant documents retrieved
        nn: total of non-relevant documents
    """
    non_hits = len(retrieved) - hits(qrels, retrieved)
    non_rel_docs = total_docs - len(qrels)
    if non_hits == 0 or non_rel_docs == 0: return 0
    return non_hits / non_rel_docs

def evaluate(model: str, coll:str, F_beta: float):
    path = os.path.join(os.path.dirname(__file__),'data')
    if model == 'vector':       path = os.path.join(path, 'relevance_vector.bin')
    elif model == 'boolean':    path = os.path.join(path, 'relevance_boolean.bin')
    elif model == 'LSI':        path = os.path.join(path, 'relevance_LSI.bin')
    else:
        print("Wrong Model")
        return
    collection = st.datasets[coll]
    collection.load_queries()
    file = open(path,'rb')
    model_relevance = pickle.load(file)
    file.close()

    precision_l = []
    recall_l = []
    f_l = []
    f1_l = []
    precision_r_l = []
    fallout_l = []
    for q in model_relevance[(model, coll)]:
        retrieved = model_relevance[(model, coll)] [q]
        qrels = collection.queries_dict[q].docs_relevance
        precision_l.append(round(precision(qrels, retrieved), 4))
        recall_l.append(round(recall(qrels, retrieved), 4))
        f_l.append(round(f_metric(qrels, retrieved, F_beta), 4))
        f1_l.append(round(f_metric(qrels, retrieved, 1), 4))
        precision_r_l.append(round(precision_ranked(qrels, retrieved), 4))
        fallout_l.append(round(fallout(qrels, retrieved, collection.numb_docs), 4))
    
    X = list(collection.queries_dict)
    plot(X, precision_l, np.average(precision_l), model + '_' + coll + '_precision', 1, Ylabel='precision')
    plot(X, recall_l,    np.average(recall_l),    model + '_' + coll + '_recall', 2, Ylabel='recall')
    plot(X, f_l,         np.average(f_l),         model + '_' + coll + '_F', 3, Ylabel='f-metric')
    plot(X, f1_l,        np.average(f1_l),        model + '_' + coll + '_F1', 4, Ylabel='f1-metric')
    plot(X, fallout_l,   np.average(fallout_l),   model + '_' + coll + '_fallout', 5, Ylabel='fallout')
    plot(X,precision_r_l,np.average(precision_r_l),model + '_' + coll + '_precision_r', 6, Ylabel='R-precision')
    
def plot(X:list, Y:list, avg: float, name:str, fig: int, Xlabel = 'queries', Ylabel=''):
    path = os.path.join(os.path.dirname(__file__), 'evaluation')
    path = os.path.join(path, name)
    n = name.split('_')
    plt.figure(fig)
    plt.scatter(X, Y )
    plt.xlabel(Xlabel)
    if Ylabel == '': Ylabel = n[2]
    plt.ylabel(Ylabel)
    plt.title(n[0]+' - '+n[1]+'  Average '+Ylabel+': '+str(round(avg,4)))
    plt.savefig(path, format='png')
    # plt.show()


# evaluate('vector', 'cranfield', 5)
# evaluate('vector', 'vaswani', 5)
# evaluate('vector', 'nfcorpus', 5)
# evaluate('LSI', 'cranfield', 5)
# evaluate('LSI', 'vaswani_r', 5)
# evaluate('LSI', 'nfcorpus_r', 5)
# evaluate('boolean', 'cranfield', 5)
# evaluate('boolean', 'vaswani', 5)
# evaluate('boolean', 'nfcorpus', 5)

# cran = st.Cranfield()     
# cran.process_docs()
# print(cran.freq_matrix.shape, cran.freq_matrix.size)      # (5510, 1400) 7,714,000
# vsw = st.Vaswani()
# vsw.process_docs()
# print(vsw.freq_matrix.shape, vsw.freq_matrix.size)        # (10179, 11429) 116,335,791
# vsw_r = st.Vaswani(reduce=True)
# vsw_r.process_docs()
# print(vsw_r.freq_matrix.shape, vsw_r.freq_matrix.size)    # (9296, 9000) 83,664,000
# nfc = st.Nfcorpus()
# nfc.process_docs()
# print(nfc.freq_matrix.shape, nfc.freq_matrix.size)        # (20420, 5371) 109,675,820
# nfc_r = st.Nfcorpus(reduce=True)
# nfc_r.process_docs()
# print(nfc_r.freq_matrix.shape, nfc_r.freq_matrix.size)    # (17354, 4000) 69,416,000