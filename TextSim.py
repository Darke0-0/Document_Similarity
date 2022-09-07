import numpy as np
import re
import nltk

"""
Text Preprocessing - Removing Stopword, Lemmatizing
"""
def utils_preprocess_text(text, flg_lemm=True, lst_stopwords=nltk.corpus.stopwords.words("english")):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

"""
Text Similarity Class
"""
class TextSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    """
    Identify the vector values for each word in the given document
    """
    def vectorize(self, text: str) -> np.ndarray:

        text = text.lower()
        words = [w for w in text.split(" ") if w not in self.stopwords]

        # normalize model
        normed_vector = self.w2v_model.get_normed_vectors()

        # creta a dict 
        w2v = dict(zip(self.w2v_model.index_to_key, normed_vector))
        word_vecs = []
        for word in words:
            try:
                vec = w2v[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        vector = np.mean(word_vecs, axis=0)
        return vector

    """
    Find the cosine similarity distance between two vectors.
    """
    def _cosine_sim(self, vecA, vecB):
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    """
    Calculates & returns similarity scores between given source document & all
    the target documents.
    """
    def calculate_similarity(self, source_doc, target_docs, threshold=0):

        source_vec = self.vectorize(source_doc)
        target_vec = self.vectorize(target_docs)
        sim_score = self._cosine_sim(source_vec, target_vec)

        if sim_score > threshold:
            results = {"similarity score": str(sim_score)}

        # Return 0 if similarity score is -ve
        else:
            results = ({"similarity score": str(0.0)})  
        return results