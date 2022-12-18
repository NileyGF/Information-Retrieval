import numpy as np
import os
import time
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import psri.structures as st


class LSI_model():
    """ Latent Semantic Indexing model for ranked information retrieval """
    
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

        try:
            self.load_TSD(self.collection.save_path)
        except:
            self.k_dim = 150  # reduced dimension
            self.Tk, self.Sk, self.DTk = self.SVD_reduced()
            np.save(os.path.join(self.collection.save_path, 'LSI_Tk'), self.Tk)
            np.save(os.path.join(self.collection.save_path, 'LSI_Sk'), self.Sk)
            np.save(os.path.join(self.collection.save_path, 'LSI_DTk'), self.DTk)

    def load_tf_idf(self,path):
        idf_f = open(os.path.join(path, 'idf_list.npy'), 'r')
        self.idf_list = np.load(os.path.join(path, 'idf_list.npy'))
        idf_f.close()

        self.tfXidf_2darray = self.Joint_tf_idf()
            
    def load_TSD(self, path):
        T_f = open(os.path.join(path, 'LSI_Tk.npy'), 'r')
        self.Tk = np.load(os.path.join(path, 'LSI_Tk.npy'))
        T_f.close()

        S_f = open(os.path.join(path, 'LSI_Sk.npy'), 'r')
        self.Sk = np.load(os.path.join(path, 'LSI_Sk.npy'))
        S_f.close()

        DT_f = open(os.path.join(path, 'LSI_DTk.npy'), 'r')
        self.DTk = np.load(os.path.join(path, 'LSI_DTk.npy'))
        DT_f.close()

        self.k_dim = self.Sk.shape[0]

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

    def SVD_reduced(self):
        
        A = self.tfXidf_2darray[ : , :self.collection.reduced]
        T, S, DT = np.linalg.svd(A, full_matrices=False)

        # dimensionality reduction:
        k = self.k_dim
        Tk = T[: , :k]
        Sk = np.diag(S[0:k])
        DTk = DT[:k , :]
        
        # add the documents using dik = Sk^-1 * Tk^T * di
        DTk_ampl = DTk
        for i in range(self.collection.reduced, self.collection.numb_docs):
            new_col = np.matmul( np.matmul (np.linalg.inv(Sk), Tk.transpose()), self.collection.freq_matrix[:,i])
            new_col = new_col.reshape(new_col.shape[0],1)
            DTk_ampl = np.append(DTk_ampl,new_col,axis=1)
            
        return Tk, Sk, DTk_ampl
    
    
    def query(self, query, ranking = 30):
        """
        Query the indexed documents using a Latent Semantic Indexing model
        """
        start_time = time.time()
        # Tokenize query
        query_tokens = self.tokenize_query(query)
        # Convert the query to the vector space
        query_vector = self.vectorize_query(query_tokens)
        # Convert the query to the reduced space
        query_reduced = self.reduce_query(query_vector)
        # Evaluate query against already processed documents
        ranked_docs = self.evaluate_query(query_reduced)
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
        filtered_text = [word for word in word_tokens if word not in LSI_model.stop_words]
        # lemmatize string
        lemmas = [LSI_model.lemmatizer.lemmatize(word, pos ='v') for word in filtered_text]
        return lemmas
            
    def vectorize_query(self, query_tokens):
        vector = np.ndarray(shape=(len(self.collection.indexed_terms)), dtype=int)

        for i in range(len(self.collection.indexed_terms)):
            freq = query_tokens.count(self.collection.indexed_terms[i])
            vector[i] = freq 
        
        return vector

    def reduce_query(self, query_freq_vector):
        """qk = Sk^-1 * Tk^T * q"""
        qk = np.matmul( np.matmul(np.linalg.inv(self.Sk), self.Tk.transpose()), query_freq_vector)
        qk = qk.reshape(qk.shape[0],1)
        return qk

    def evaluate_query(self, qk_vector):
        """
        Evaluates the query against the corpus
        :param query_tokens: list of query tokens
        :returns: list of matching documents
        """
        doc_likehood = {}
        q_norm = np.linalg.norm(qk_vector)
        for i in range(len(self.collection.documents_list)):
            di_x_q = np.dot(self.DTk[:,i], qk_vector)
            di_norm = np.linalg.norm(self.tfXidf_2darray[:,i])
            norm_prod = di_norm * q_norm
            if di_x_q == 0 or norm_prod == 0:
                doc_likehood[i+1] = 0
            else:
                doc_likehood[i+1] = di_x_q / norm_prod
        
        ranked_doc = dict(sorted(doc_likehood.items(), key=lambda item: item[1], reverse=True))
        return ranked_doc

