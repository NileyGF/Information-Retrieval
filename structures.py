import ir_datasets
import os
import numpy as np
import pickle
import time
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Document:
    """ Represents a document across the collections """
    def __init__(self, id:int, text:str, title=''):
        self.id = id
        self.text = text
        self.title = title

        # it's indexed terms
        self.terms_list = [] 
    
class Query:
    """ Represents a query across the collections """
    def __init__(self, id:int, query:str):
        self.id = id
        self.query = query
        # the relevant documents for this query
        self.docs_relevance = []


class Std_tokenizer:
    """ This class contains the common behavior among the collections regarding documents processing """
    
    stop_words = list(set(stopwords.words("english")))  # nltk stopwords
    lemmatizer = WordNetLemmatizer()                    # nltk lemmatizer
    
    def docs_process(docs_list):
        """ 
        Receive a list of Document and tokenize each one.
        Returns a list of Document, now with the indexed terms 
         for every document; and a list with all the indexed terms.
        """
        start_time = time.time()

        # list of all indexed terms
        indexed_terms = []

        for i in range(len(docs_list)):
            doc_tokens = Std_tokenizer.tokenize_nltk(docs_list[i].text)
            docs_list[i].terms_list = doc_tokens    # assign the terms of the Document i
            for x in doc_tokens:         
                # update the indexed terms list       
                indexed_terms.append(x)        
        
        unique_words = list(set(indexed_terms))  # remove repeated terms by using set()
        unique_words.sort()                      # sort to have a unique order when indexing

        return docs_list, unique_words
    
    def tokenize_nltk(doc_text):
        text = doc_text.lower()
        # remove numbers
        text = text.translate(str.maketrans('', '', string.digits))
        # remove punctuation (replace them with ' ')
        text = text.translate(str.maketrans(Collection.punctuations(), ' '*len(Collection.punctuations())))
        # generate tokens
        word_tokens = word_tokenize(text)
        # remove stopwords 
        filtered_text = [word for word in word_tokens if word not in Std_tokenizer.stop_words]
        # lemmatize words in the list of tokenized words
        lemmas = [Std_tokenizer.lemmatizer.lemmatize(word, pos ='v') for word in filtered_text]
        return lemmas
    
    def structure_data(docs_list, terms):
        start_time = time.time()
        # take the documents and the indexed terms to create the inverted terms, and matrix representations
        docs_by_term_dict = Std_tokenizer.dictionary_strctr(terms, docs_list, start_time)
        term_doc_matrix = Std_tokenizer.freq_m_strctr(terms, docs_list, docs_by_term_dict, start_time)
        
        return docs_by_term_dict, term_doc_matrix

    def dictionary_strctr(terms, docs_list, start_time):
        # a dictionary term:list . For every term in the collection, the documents it's in
        docs_by_term_dict = {t: [] for t in terms}

        for i in range(len(docs_list)):
            no_repeat = set(docs_list[i].terms_list)
            #for every unique term in the document, add to dict[term] this document.id
            for t in no_repeat:
                docs_by_term_dict[t].append(docs_list[i].id)

        return docs_by_term_dict

    def freq_m_strctr(terms, docs_list, docs_by_term_dict, start_time):
        """ A more efficient approach than exploring the whole matrix by rows and columns """
        term_doc_matrix = np.ndarray((len(terms),len(docs_list)), dtype=int)
        # fill the non-zero positions of the frequency matrix 
        for i in range(len(terms)):
            for k in range(len(docs_by_term_dict[terms[i]])):  
                # only count the frequency of a term for the documents it's in  
                d_ind = docs_by_term_dict[terms[i]] [k]  - 1   # minus 1, because the document's id is 1-indexed
                freq = docs_list[d_ind].terms_list.count(terms[i])
                term_doc_matrix[i,d_ind] = freq

        return term_doc_matrix

    def pickle_dump(path, docs_list, docs_by_term_dict, terms, term_doc_matrix):
        # save the processed data to binary files, using pickle. 
        # this allow us to read later the file as a python structure

        # we're not saving the frequency matrix because of its big weigth. (reaching 2GB)
        # We rather save the lists and dictionaries and recreate the matrix in less than 10 seconds 
        dimensions_f = open(os.path.join(path, 'dimensions.bin'),'wb')
        pickle.dump(term_doc_matrix.shape, dimensions_f)
        dimensions_f.close()

        docs_by_term_f = open(os.path.join(path, 'docs_by_term_dict.bin'),'wb')
        pickle.dump(docs_by_term_dict, docs_by_term_f)
        docs_by_term_f.close()

        docs_f = open(os.path.join(path, 'docs_list.bin'),'wb')
        pickle.dump(docs_list, docs_f)
        docs_f.close()

        unique_terms_f = open(os.path.join(path, 'unique_terms.bin'),'wb')
        pickle.dump(terms, unique_terms_f)
        unique_terms_f.close()

class Collection:
    """ Abstract class Collection, with the common behavior of the collections"""
    def punctuations(boolean=False):
        #in the boolean model we don't remove '&!|' in the same way as in the other models
        if boolean: return "\"#$%'()*+,-./:;<=>?@[]\\^_{`}~" 
        return "!\"#$%&'()*+,-./:;<=>?@[]\\^_`{|}~" 

    def load_files(self):
        # try to load the saved data
        path = self.save_path
        try:
            docs_f = open(os.path.join(path, 'docs_list.bin'),'rb')
            self.documents_list = pickle.load(docs_f)
            docs_f.close()

            docs_by_term_f = open(os.path.join(path, 'docs_by_term_dict.bin'),'rb')
            self.terms_dict = pickle.load(docs_by_term_f)
            docs_by_term_f.close()

            unique_terms_f = open(os.path.join(path, 'unique_terms.bin'),'rb')
            self.indexed_terms = pickle.load(unique_terms_f)
            unique_terms_f.close()

            self.freq_matrix = Std_tokenizer.freq_m_strctr(self.indexed_terms, self.documents_list, self.terms_dict, time.time())
            
            self.loaded_metadata = True
        except:
            self.loaded_metadata = False
    
    def docs_ranking(self, ranking, docs_id_list):
        # create a list with the information of the top ranking documents in docs_id_list
        result = []   #tuple list 
        if not self.loaded_docs and not self.loaded_metadata:
            self.load_docs()
        docs_id_list = docs_id_list[:ranking]
        for id in docs_id_list:
            for doc in self.documents_list:
                if id == doc.id: 
                    result.append((doc.id, doc.title, doc.text))
                    break
        
        return result[:ranking]

class Cranfield(Collection):
    def __init__(self):
        self.loaded_docs = False
        self.loaded_queries = False
        self.loaded_rel = False
        self.loaded_metadata = False
        self.reduced = 1400         # ammount of documents for the LSI
        self.numb_docs = 1400        
        self.save_path = os.path.join(os.path.join(os.path.dirname(__file__),'data'), 'cranfield_treated')
        self._generator = ir_datasets.load('cranfield')

    def load_docs(self):
        self.documents_list = []
        i = 0
        for doc in self._generator.docs_iter():
            if i >= self.numb_docs: # for reduced collections
                break
            self.documents_list.append(Document(int(doc.doc_id), doc.text, doc.title))
            i+=1
        self.loaded_docs = True

    def load_queries(self):
        # a dictionary of Query
        self.queries_dict = {}
        # load queries
        for query in self._generator.queries_iter():
            self.queries_dict[int(query.query_id)] = Query(int(query.query_id), query.text)
        self.loaded_queries = True
        # load queries relevances
        for qrel in self._generator.qrels_iter():
            if int(qrel.doc_id) <= self.numb_docs:
                q = self.queries_dict.get(int(qrel.query_id))
                if q: #if the qrel.query_id is in self.queries_dict
                    q.docs_relevance.append(int(qrel.doc_id))
        self.loaded_rel = True

    def process_docs(self):
        # tokenize and lemmatize the documents
        if not self.loaded_docs: 
            self.load_docs()
        self.documents_list , self.indexed_terms = Std_tokenizer.docs_process(self.documents_list)
        self.terms_dict, self.freq_matrix = Std_tokenizer.structure_data(self.documents_list, self.indexed_terms)
        # save to files
        Std_tokenizer.pickle_dump(self.save_path, self.documents_list, self.terms_dict, self.indexed_terms, self.freq_matrix)

class Vaswani(Collection):
    def __init__(self, reduce = False):
        self.loaded_docs = False
        self.loaded_queries = False
        self.loaded_rel = False
        self.loaded_metadata = False
        if reduce: # reduce the dataset taking less documents, disminishing the matrix dimmensions
            self.numb_docs = 9000 
            self.save_path = os.path.join(os.path.join(os.path.dirname(__file__),'data'), 'vaswani_r_treated')
        else:
            self.numb_docs = 11429
            self.save_path = os.path.join(os.path.join(os.path.dirname(__file__),'data'), 'vaswani_treated')

        self.reduced = 4000     # ammount of documents for the LSI, cause the matrix is still too big
        self._generator = ir_datasets.load('vaswani')

    def load_docs(self):
        self.documents_list = []
        i = 0
        for doc in self._generator.docs_iter():
            if i >= self.numb_docs: # if reduce = True
                break
            self.documents_list.append(Document(int(doc.doc_id), doc.text, doc.text[:60] + '...'))
            i+=1
        self.loaded_docs = True

    def load_queries(self):
        # a dictionary of Query
        self.queries_dict = {}
        # load queries
        for query in self._generator.queries_iter():
            self.queries_dict[int(query.query_id)] = Query(int(query.query_id), query.text)
        self.loaded_queries = True
        # load queries relevances
        for qrel in self._generator.qrels_iter():
            if int(qrel.doc_id) <= self.numb_docs:
                q = self.queries_dict.get(int(qrel.query_id))
                if q: # if the qrel.query_id is in self.queries_dict
                    q.docs_relevance.append(int(qrel.doc_id))
        self.loaded_rel = True

    def process_docs(self):
        # tokenize and lemmatize the documents
        if not self.loaded_docs: 
            self.load_docs()
        self.documents_list , self.indexed_terms = Std_tokenizer.docs_process(self.documents_list)
        self.terms_dict, self.freq_matrix = Std_tokenizer.structure_data(self.documents_list, self.indexed_terms)
        # save to files
        Std_tokenizer.pickle_dump(self.save_path, self.documents_list, self.terms_dict, self.indexed_terms, self.freq_matrix)

class Nfcorpus(Collection):
    def __init__(self, reduce=False):
        self.loaded_docs = False
        self.loaded_queries = False
        self.loaded_rel = False
        self.loaded_metadata = False
        if reduce: # reduce the dataset taking less documents, disminishing the matrix dimmensions
            self.numb_docs = 4000
            self.save_path = os.path.join(os.path.join(os.path.dirname(__file__),'data'), 'nfcorpus_r_treated')
        else:
            self.numb_docs = 5371
            self.save_path = os.path.join(os.path.join(os.path.dirname(__file__),'data'), 'nfcorpus_treated')
        
        self.reduced = 2500     # ammount of documents for the LSI, cause the matrix is still too big
        self._generator = ir_datasets.load('nfcorpus/dev')

    def load_docs(self):
        self.documents_list = []
        i = 0
        for doc in self._generator.docs_iter():
            if i >= self.numb_docs:
                break

            text_encode = doc.abstract.encode("ascii","replace")
            text_decode = text_encode.decode()
            title_encode = doc.title.encode("ascii","replace")
            title_decode = title_encode.decode()
            self.documents_list.append(Document(int(doc.doc_id[4:]), text_decode, title_decode))
            i+=1
        self.loaded_docs = True

    def load_queries(self):
        # a dictionary of Query
        self.queries_dict = {}
        # load queries
        for query in self._generator.queries_iter():
            #remove unicode characters
            q_encode = (query.title.encode("ascii","replace")).decode()
            self.queries_dict[int(query.query_id[6:])] = Query(int(query.query_id[6:]), q_encode)
        self.loaded_queries = True
        # load queries relevances
        for qrel in self._generator.qrels_iter():
            if int(qrel.doc_id[4:]) <= self.numb_docs:
                q = self.queries_dict.get(int(qrel.query_id[6:]))
                if q: # if the qrel.query_id is in self.queries_dict
                    q.docs_relevance.append(int(qrel.doc_id[4:]))
        self.loaded_rel = True

    def process_docs(self):
        # tokenize and lemmatize the documents
        if not self.loaded_docs: 
            self.load_docs()
        self.documents_list , self.indexed_terms = Std_tokenizer.docs_process(self.documents_list)
        self.terms_dict, self.freq_matrix = Std_tokenizer.structure_data(self.documents_list, self.indexed_terms)
        # save to files
        Std_tokenizer.pickle_dump(self.save_path, self.documents_list, self.terms_dict, self.indexed_terms, self.freq_matrix)

class Bool_node:
    # boolean model query node, positive
    def __init__(self, lex) -> None:
        self.lex = str(lex)

    def node_type(self):
        return 'basic'
    
    def node_value(self):
        return self.lex

    def __str__(self) -> str:
        return self.lexs

class Bool_Not_node:
    # boolean model query node, negated
    def __init__(self, term):
        self.lex = str(term)

    def node_type(self):
        return 'not'
    
    def node_value(self):
        return self.lex

    def __str__(self) -> str:
        return self.lex

datasets = {
    'cranfield'   : Cranfield(),
    'cranfield_r' : Cranfield(),
    'vaswani'     : Vaswani(),
    'vaswani_r'   : Vaswani(reduce=True),
    'nfcorpus'    : Nfcorpus(),
    'nfcorpus_r'  : Nfcorpus(reduce=True)
}
