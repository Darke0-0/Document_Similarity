#  Model for creating word embeddings

# Importing Libraries
import re
import nltk
import gensim
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

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
init callback class
"""
class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    # Printing loss at the end of every 10 epoch
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        elif self.epoch%10 == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

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
        results = []
        target_vec = self.vectorize(target_docs)
        sim_score = self._cosine_sim(source_vec, target_vec)
        if sim_score > threshold:
            results.append({"score": sim_score})
        else:
            results.append({"score": 0.0})  

        return results

"""
Loading the model
"""
model_path = 'Final_model.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path,binary=True)
model = TextSim(w2v_model)


if __name__ == '__main__':
    """
    Adding argument for train/test call
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='test')
    args = parser.parse_args()

#  Training Call
if args.train == 'train':
    df = pd.read_csv("Precily_Text_Similarity.csv")
    df["text1_clean"] = df["text1"].apply(lambda x: utils_preprocess_text(x, flg_lemm=True))
    df["text2_clean"] = df["text2"].apply(lambda x: utils_preprocess_text(x, flg_lemm=True))

    texts = list(df['text1_clean']) + list(df['text2_clean'])
    # tokenize
    c = 0
    for text in tqdm(texts):
        texts[c] = list(gensim.utils.tokenize(text, deacc=True, lower=True))
        c += 1
    
    # train model
    w2v = gensim.models.Word2Vec(texts, vector_size=300,window=10,min_count=1,epochs=100,negative=20,sample=1e-4,callbacks=[callback()],compute_loss=True)
    w2v.wv.save_word2vec_format('Final_model.bin',binary=True)

# Testing Call
elif args.train == 'test':
    """
    Taking and processing input
    """
    prep_s1 = utils_preprocess_text(input('Enter Statement 1 \n'), flg_lemm=True)
    prep_s2 = utils_preprocess_text(input('Enter Statement 2 \n'), flg_lemm=True)

    print('Semantic similarity :-',model.calculate_similarity(prep_s1, prep_s2)[0]['score'])
