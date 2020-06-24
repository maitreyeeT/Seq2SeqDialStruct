# Note: you're note allowed to import any other library.
from __future__ import print_function
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import multiprocessing

def train_word2vec(sentence_matrix,num_features=300, min_word_count=1, context=10, root_dir=None):
    """
    # TODOs:
    1. Reload word2vec model if exists. If it doesn't, save it with the name as 300features_1minwords_10context.
    2. Train word2vec model if doens't exist.
    3. Output: returns model and initial weights for embedding layer.
    4. Input params:
        sentence_matrix # int matrix: num_sentences x max_sentence_len
        vocabulary_inv  # dict {int: str}
        num_features    # Word vector dimensionality                      
        min_word_count  # Minimum word count                        
        context         # Context window size 
    """

    embedding_weights = {}
    embedding_model = None
    # Note: Please use model_path to save pre-trained embedding.
    if root_dir:
        model_path = os.path.join(root_dir, "trained_models/300feats_1minwords_10context")
    else:
        model_path = "../trained_models/300feats_1minwords_10context"
    model = word2vec.Word2Vec(size=num_features, window=context, min_count=min_word_count, workers=4)
    model.build_vocab(sentence_matrix, progress_per=1000)
    model.train(sentence_matrix, total_examples=model.corpus_count, epochs=50, report_delay=1)
    print(model.wv.most_similar(positive=['yeah']))

    model.save(model_path)
    #embedding weights
    embedding_model = word2vec.\
        Word2Vec.load('/home/maitreyee/Development/'
                      'autoencoder/trained_models/'
                      '300feats_2minwords_5context')
    embedding_weights = embedding_model.wv.vectors

    return embedding_model, embedding_weights


def data_transform():
    import pandas as pd
    data = pd.read_csv('/home/maitreyee/Development'
                       '/autoencoder/Dialog_bankAll_withSynDA3.csv', sep ='\t')
    col = []
    def build_corpus(data):
        #Creates a list of lists containing words from each sentence
        corpus = []
        for sentence in data.astype(str).\
            str.lower().str.replace(r'[^\s\w]','',regex=True)\
            .str.strip(' ').iteritems():
            word_list = sentence[1].split(' ')
            corpus.append(word_list)
        return corpus

    col = build_corpus(data.utterances)

    return col

if __name__ == '__main__':
    data_embed = data_transform()
    print(data_embed[0:10])
    model = train_word2vec(data_embed)