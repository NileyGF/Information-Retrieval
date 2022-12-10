import numpy as np
import time
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import structures as st

class Boolean_model():
    """ Boolean model for unranked information retrieval """
    # or -> ||
    # and -> &&
    # not -> !!
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

        # Print the time it took to read the collection data
        print("First load time:", round(time.time() - self.start_time, 2), "seconds")
        # cranfield    0.91 sec
        # vaswani      2.41 sec
        # nfcorpus     6.72 sec

    def query(self, query_text, ranking=30):
        """
        Query the indexed documents using a boolean model
        """
        start_time = time.time()

        # Tokenize query
        query_tokens = self.tokenize_query(query_text)
        if len(query_tokens) == 0:
            return []        
        if '&&' in query_tokens or '||' in query_tokens or '!!' in query_tokens:
            # Parse the query
            query_vector = self.parse_query(query_tokens)
            # Evaluate query against already processed documents as a boolean query
            ranked_docs = self.evaluate_bool_query(query_vector)
        else:
            query_tf = self.tf_query(query_tokens)
            # Evaluate query against already processed documents as a special query
            ranked_docs = self.evaluate_query(query_tokens, query_tf)

        # Return only non-0-relevance docs
        i = 0
        while i < len(ranked_docs.values()) and list(ranked_docs.values())[i] > 0:
            i+=1
            if i >=ranking: break
        if i < ranking: ranking = i
        
        index_list = list(ranked_docs.keys())[0:ranking]
        docs_to_print = self.collection.docs_ranking(ranking, index_list)
        # Print the time it took to gather the tokens for the collection
        print("Query tokenization time:", round(time.time() - start_time, 2), "seconds")
        # cranfield    0.04 sec
        # vaswani      0.35 sec
        # nfcorpus     0.11 sec
        return docs_to_print

    def tokenize_query(self, query):
        """
        Preprocesses the query given as input. 
        Converts to lower case, removes the punctuations, splits on whitespaces and removes stopwords.
        """
        # q = "t1 && t2 && t3 || t1 && t2 && !! t4"

        text = query.lower()
        # Remove numbers
        text = text.translate(str.maketrans('', '', string.digits))
        # remove punctuation
        text = text.translate(str.maketrans(st.Collection.punctuations(boolean=True), ' '*len(st.Collection.punctuations(boolean=True))))
        # parse the !, &, |
        text = self.check_punct(text)
        # split on whitespaces to generate tokens
        word_tokens = text.split()
        # remove stopwords function
        filtered_text = [word for word in word_tokens if word not in Boolean_model.stop_words]
        # lemmatize string
        lemmas = [Boolean_model.lemmatizer.lemmatize(word, pos ='v') for word in filtered_text]
        return lemmas
    
    def check_punct(self, text):
        result = ''
        i=0
        while i < len(text) and i>=0:
            if i == len(text) - 1:
                if text[i] == '!' or text[i] == '&' or text[i] == '|':
                    result += ' '
                else:result += text[i] 
            else:  
                if text[i] == '!':
                    if not text[i+1] == '!':
                        result += ' '
                    else: 
                        result += '!!'
                        i+=1
                elif text[i] == '&':
                    if not text[i+1] == '&':
                        result += ' '
                    else: 
                        result += '&&'
                        i+=1
                elif text[i] == '|':
                    if not text[i+1] == '|':
                        result += ' '
                    else: 
                        result += '||'
                        i+=1
                else:
                    result += text[i]            
            i+=1
        return result

    def tf_query(self, query_tokens):
        unique = list(set(query_tokens))
        tf_dict = {}
        for i in range(len(unique)):
            tf_dict[unique[i]] = query_tokens.count(unique[i])
        max_freq = np.array(list(tf_dict.values())).max()
        if max_freq == 0: 
            return tf_dict
        for i in tf_dict:
            tf_dict[i] = tf_dict[i] / max_freq
        return tf_dict


    def parse_query(self, query_tokens):
        vector = []
        cc = []

        # parsing the query into a list of conjunctive components
        i=0
        while i < len(query_tokens) and i>=0:
            if i == 0:
                if query_tokens[i] == '&&' or query_tokens[i] == '||'  or query_tokens[i] == '!!' :
                    print('wrong query')
                    return
                cc.append(st.Bool_node(query_tokens[i]))
                if (i == len(query_tokens) - 1):
                    vector.append(cc)
                    cc =[]
                i+=1
            elif (i == len(query_tokens) - 1):
                if query_tokens[i] == '&&' or query_tokens[i] == '||'  or query_tokens[i] == '!!' :
                    print('wrong query')
                    return
                cc.append(st.Bool_node(query_tokens[i]))
                vector.append(cc)
                cc = []
                i+=1
            else:
                if query_tokens[i] == '||':
                    if query_tokens[i+1] == '&&' or query_tokens[i+1] == '||':
                        print('wrong query')
                        return
                    vector.append(cc)
                    cc = []
                elif query_tokens[i] == '!!':
                    if query_tokens[i+1] == '&&' or query_tokens[i+1] == '||'  or query_tokens[i+1] == '!!' :
                        print('wrong query')
                        return
                    cc.append(st.Bool_Not_node(query_tokens[i+1]))
                    i+=1
                elif query_tokens[i] == '&&':
                    if query_tokens[i+1] == '&&' or query_tokens[i+1] == '||':
                        print('wrong query')
                        return
                else:
                    cc.append(st.Bool_node(query_tokens[i]))
                i+=1
        return vector

    def evaluate_bool_query(self, query_cc_list, query_tf):
        """
        Evaluates the query against the corpus
        """
        doc_likehood = {d.id:0 for d in self.collection.documents_list}
        
        for d in range(1,len(self.collection.documents_list)+1):
            # for every document check if it has any whole conjunctive component
            for cc in query_cc_list:
                rel = True
                for t in cc:
                    ds = self.collection.terms_dict.get(t.node_value())
                    if ds:
                        if type(t) == st.Bool_Not_node:     # negative literal
                            if d in ds:
                                rel = False
                        else:                               # positive literal
                            if not d in ds:
                                rel = False
                    else: # term not in collection
                        rel = False
                    if not rel:
                        break # if it lacks a term, it won't be relevant ( 0 and x = 0)
                doc_likehood[d] = int(rel)  #if True:1, if False:0
                if rel: 
                    break # if it's already relevant don't check the others cc ( 1 or x = 1)
               
        ranked_doc = dict(sorted(doc_likehood.items(), key=lambda item: item[1], reverse=True))
        return ranked_doc

    def evaluate_query(self, query_tokens, query_tf):
        """
        Evaluates the query against the corpus
        """
        doc_likehood = {d.id:0 for d in self.collection.documents_list}
        for t in query_tokens:            # intersection of relevant documents per term
            term_value = query_tf.get(t)
            if not term_value: term_value = 0
            ds = self.collection.terms_dict.get(t)
            if ds:
                for d in ds: 
                    doc_likehood[d] += term_value
        
        ranked_doc = dict(sorted(doc_likehood.items(), key=lambda item: item[1], reverse=True))
        return ranked_doc
