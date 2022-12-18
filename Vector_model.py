import numpy as np
import os
import time
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import psri.structures as st

class Vector_model():
    """ Vector space model for ranked information retrieval """

    stop_words = list(set(stopwords.words("english")))  # nltk stopwords
    lemmatizer = WordNetLemmatizer()                    # nltk lemmatizer
    lemmatizer.lemmatize('', pos ='v')                  # initialize the lemmatizer (because of the lazy load)
    
    def __init__(self, collection= 'cranfield'):
        self.start_time = time.time() 
        self.collection = st.datasets[collection]

        self.collection.load_files()
        if not self.collection.loaded_metadata:
            self.collection.process_docs()
            self.collection.load_files()

        try:
            self.load_tf_idf(self.collection.save_path)
        except:
            self.idf_list = self.idf()
            self.tfXidf_2darray = self.Joint_tf_idf() 
            np.save(os.path.join(self.collection.save_path, 'idf_list'), self.idf_list)


    def load_tf_idf(self, path):
        idf_f = open(os.path.join(path, 'idf_list.npy'), 'r')
        self.idf_list = np.load(os.path.join(path, 'idf_list.npy'))
        idf_f.close()

        self.tfXidf_2darray = self.Joint_tf_idf()   

    def idf(self):
        """
        Calculates the inverse document frequency of every term.
        idf[i] = log(total_docs / number of docs where is the term i)
        """
        total_docs = len(self.collection.documents_list)
        
        idf = []
        for term in self.collection.terms_dict:
            idf.append(np.log10(float(total_docs / len(self.collection.terms_dict[term]))))

        return idf

    def Joint_tf_idf(self):
        """
        Calculates the TF*IDF of every term.
        tf[i,d] = freq[i,d] / max freq[d]
        tfxidf[i,d] = tf[i,d] * idf[i]
        """
        max_freq = self.collection.freq_matrix.max(axis=0, keepdims=True)
        tf_x_idf = np.ndarray(self.collection.freq_matrix.shape, dtype=float)
        terms = self.collection.indexed_terms
        t_dict = self.collection.terms_dict

        # fill the non-zero positions       
        for i in range(len(terms)):
            for k in range(len(t_dict[terms[i]])):
                d_ind = t_dict[terms[i]] [k] - 1 # minus 1, because the document's id is 1-indexed
                
                tf_i_d = self.collection.freq_matrix[i,d_ind] / max_freq[0,d_ind]
                tf_x_idf[i,d_ind] = tf_i_d * self.idf_list[i]

        return tf_x_idf

    def query(self, query_text, ranking = 30):
        """
        Query the indexed documents using a vector space model
        query: valid expression to search for
        returns: top-ranking relevant documents
        """
        start_time = time.time()

        # Tokenize query
        query_tokens = self.tokenize_query(query_text)
        # Convert the query to the vector space
        query_vector = self.vectorize_query(query_tokens)
        # Weight of terms in the query
        query_weight = self.weight_query(query_vector)
        # Evaluate query against already processed documents
        ranked_docs = self.evaluate_query(query_weight)
        # Return only non-0-relevance docs
        i = 0
        while list(ranked_docs.values())[i] > 0:
            i+=1
            if i >=ranking: break
        if i < ranking: ranking = i

        index_list = list(ranked_docs.keys())[0:ranking]
        docs_to_print = self.collection.docs_ranking(ranking, index_list)

        return docs_to_print

    def tokenize_query(self, query):
        """
        Preprocesses the query given as input. 
        Converts to lower case, removes the punctuations, splits on whitespaces and removes stopwords.
        """
        text = query.lower()
        # Remove numbers
        text = text.translate(str.maketrans('', '', string.digits))
        # remove punctuation
        text = text.translate(str.maketrans(st.Collection.punctuations(), ' '*len(st.Collection.punctuations())))
        # split on whitespaces to generate tokens
        word_tokens = word_tokenize(text)
        # remove stopwords function
        filtered_text = [word for word in word_tokens if word not in Vector_model.stop_words]
        # lemmatize string
        lemmas = [Vector_model.lemmatizer.lemmatize(word, pos ='v') for word in filtered_text]
        return lemmas
        
    def vectorize_query(self, query_tokens):
        vector = np.ndarray(shape=(len(self.collection.indexed_terms)), dtype=int)

        for i in range(len(self.collection.indexed_terms)):
            freq = query_tokens.count(self.collection.indexed_terms[i])
            vector[i] = freq 
        
        return vector

    def weight_query(self, query_freq_vector, softer=0.1):
        weight = np.ndarray(shape=(len(self.collection.indexed_terms)), dtype=float)
        max_freq = query_freq_vector.max()
        if max_freq - 0 < 1e-10 :
            return weight
        for i in range((len(self.collection.indexed_terms))):
            if query_freq_vector[i] == 0:
                wiq = softer * self.idf_list[i]
            else:
                wiq = ( softer + (1 - softer) * (query_freq_vector[i] / max_freq) ) * self.idf_list[i]
            weight[i] = wiq
        
        return weight

    def evaluate_query(self, query_weight_vector):
        """
        Evaluates the query against the corpus
        :param query_tokens: list of query tokens        :param query_tokens: list of query tokens

        :returns: list of matching documents
        """
        doc_likehood = {}
        q_norm = np.linalg.norm(query_weight_vector)
        for k in range((len(self.collection.documents_list))):
            dk_x_q = np.dot(self.tfXidf_2darray[:,k],query_weight_vector)
            dk_norm = np.linalg.norm(self.tfXidf_2darray[:,k])
            norm_prod = dk_norm * q_norm
            if dk_x_q == 0 or norm_prod == 0:
                doc_likehood[k+1] = 0
            else:
                doc_likehood[k+1] = dk_x_q / norm_prod
        
        ranked_doc = dict(sorted(doc_likehood.items(), key=lambda item: item[1], reverse=True))
        return ranked_doc


