import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def assemble_corpus(data):
    ''' assembles corpus (list of docs) from dataframe
        param
        -----
        data                dataframe, must have 'tokens' feature as list of strings

        return
        ------
        corpus              list, each doc a string
        corpus_tokenized    list, each doc a list of stings
    '''
    chapter_names = list(data.chapter_name)
    chapter_docs = dict(zip(chapter_names, data.tokens))
    corpus_tokenized = list(chapter_docs.values()) # each doc is list of tokens
    corpus = [] # each doc one string
    for doc in corpus_tokenized:
        corpus.append(' '.join([x for x in doc]))

    return corpus, corpus_tokenized

def tfidf_top_words(corpus, tfidf_vectorizer, chapter_names, n):
    '''
    get top n words for a document from the tfidf matrix
    map the doc's tfidf vector to the feature names (words)
    return a list of tuples of the top n words, sorted by tfidf score

    param
    -----
    corpus                  list, of docs (each one string)
    tfidf_vectorizer        fitted sklearn estimator
    chapter_names           list, chapter titles as strings, len=len(corpus)
    n                       int

    return
    ------
    chapter_top_words       dict, {chapter: [(word,score)]} for n words, sorted
    '''
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    feat_names = tfidf_vectorizer.get_feature_names()
    chapter_top_words = {}
    for i, name in enumerate(chapter_names):
        tfidf_vector = tfidf[i,:].toarray().flatten()
        # get indices of top n
        ii = np.argsort(tfidf_vector)[-n -1 : -1]
        # get values
        words = [feat_names[i] for i in ii]
        scores = [np.round(tfidf_vector[i], 4) for i in ii]
        # sort descending
        output = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
        chapter_top_words[name] = output

    return chapter_top_words

def tfidf_top_words_fitted(corpus, tfidf_vectorizer, chapter_names, n):
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    feat_names = tfidf_vectorizer.get_feature_names()
    chapter_top_words = {}
    for name, document in zip(chapter_names, corpus):
        tfidf_vector = tfidf_vectorizer.transform([document])
        vector = tfidf_vector.toarray().flatten()
        ii = np.argsort(vector)[-n:-1]
        words = [tfidf_feat_names[i] for i in ii]
        chapter_top_words[name] = words

    return chapter_top_words

def main(pkl_file, n_keywords):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # assemble corpus
    corpus, corpus_tokenized = assemble_corpus(data)

    # tf-idf and tf vectors
    n_features = 1000

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feat_names = tfidf_vectorizer.get_feature_names()

    print('bag-of-words matrix dimensions: {} documents x {} words'.format(tfidf.shape[0], tfidf.shape[1]))

    # keyword extraction from TF-IDF for each chapter
    chapter_names = list(data.chapter_name)
    top_words_tfidf = tfidf_top_words(corpus, tfidf_vectorizer, chapter_names, n=n_keywords)
    data['keywords'] = data.chapter_name.apply(lambda x: top_words_tfidf[x])

    outfile = f'pkl/data_keywords_{n_keywords}.pkl'
    with open(outfile, 'wb') as f:
       pickle.dump(data, f)
    print(f'file written: {outfile}')

if __name__=='__main__':
    ''' Adds 'keywords' feature to dataframe with 'tokens'
        keywords selected as top n words in tfidf vector for each doc in corpus

        cmd line args
        -------------
        [1] pkl_file    pandas df, cleaned dataframe
        [2] n_keywords  int, n keywords to pick

        output
        ------
        pkl file        pandas df with feature 'keywords'
                        filename = pkl/data_keywords_<n_keywords>.pkl'

    '''
    main(sys.argv[1], int(sys.argv[2]))
