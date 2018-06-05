from __future__ import print_function
from collections import Counter
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

dic_words = {}


def combine_together(file):
    for line in open(file).readlines():
        items = line.strip().split(',')
        key = items[1]
        if key in dic_words:
            dic_words[key] += 1
        else:
            dic_words[key] = 1


dic_words["<PAD/>"] = 100


def build_vocab(word_dic):
    word_counts = Counter(word_dic)
    vocabulary_inv_list = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv_list)}
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    return [vocabulary, vocabulary_inv]


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=50, min_word_count=1, context=10):
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')
        embedding_model = word2vec.Word2Vec(sentence_matrix, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)
        embedding_model.init_sims(replace=True)
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}

    return embedding_model, embedding_weights


if __name__ == '__main__':
    # put all produc_Id and diag_cd in dictionary dic_words by the occurence of each word
    # combine_together("hae_dx_new.csv")
    # combine_together("hae_rx.csv")
    # combine_together("nonhae_dx.csv")
    # combine_together("nonhae_rx.csv")
    combine_together("./data/nonhae_sorted_feature_extraction")
    combine_together("../data/hae_feature_extraction")

    # build word2vec model
    vocabulary, vocabulary_inv = build_vocab(dic_words)
    sentences = [[vocabulary_inv[w]] for w in vocabulary_inv.keys()]
    embedding_model_result, embedding_weights_result = train_word2vec(sentences, vocabulary_inv)
    print(embedding_model_result['V2389'])





