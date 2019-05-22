
import pandas as pd
import numpy as np

# Spacy used primarily for lemmatization
import spacy

# Gensim for Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Import stopwords from NLTK
from nltk.corpus import stopwords

from pprint import pprint

class GensimTopicModeler():
    '''
    Tokenizes corpus by removing stopwords, creating bigrams, and lemmatizing words.
    Uses the Gensim library's application of Latent Dirichlet Allocation to model 
    topics within the Harry Potter corpus.
    The number of topics can be optimized using Gensim's coherence score.
    '''
    def __init__(self, corpus, verbose=False):
        self.corpus = corpus
        self.verbose = verbose

    def _remove_stopwords(self, corpus):
        stop_words = stopwords.words('english')
        stop_words.extend(['not', 'look', 'do', 'go', 'get', 'would', 'be', 's', 'say',
                           'see', 'could', 'back', 'know', 'come', 
                           'harry', 'hermione',
                           'think', 'tell', 'take', 'make', 'want'])

        return [[word for word in doc if word not in stop_words] for doc in corpus]
        
    def _create_bigrams(self, corpus):
        if self.verbose == True:
            print("Building bigrams from corpus...")

        # Create bigram model from corpus
        bigram = gensim.models.Phrases(self.corpus, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        return [bigram_mod[doc] for doc in self.corpus]

    def _lemmatization(self, corpus, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        if self.verbose == True:
            print("Lemmatizing corpus...")

        nlp = spacy.load('en', disable=['parser', 'ner'])
        lemma_corpus= []
        for sent in corpus:
            doc = nlp(" ".join(sent)) 
            lemma_corpus.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return lemma_corpus

    def tokenize_corpus(self):
        # Remove stopwords from corpus
        self.corpus = self._remove_stopwords(self.corpus)

        # Build bigram model from corpus
        self.corpus = self._create_bigrams(self.corpus)

        # Lemmatize corpus
        self.corpus = self._lemmatization(self.corpus)

        # Remove stopwords that snuck through after lemmatizing
        self.corpus = self._remove_stopwords(self.corpus)

    def fit(self, num_topics):
        #Create dictionary of tokens and tf matrix, then fit LDA model

        # Create Dictionary
        self.id2word = corpora.Dictionary(self.corpus)

        # Term Document Frequency (bag of words)
        self.tdf = [self.id2word.doc2bow(text) for text in self.corpus]

        # Build LDA model
        if self.verbose == True:
            print('Creating LDA model...')
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.tdf,
                                                id2word=self.id2word,
                                                num_topics=num_topics, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

        # Print the Keyword in the 10 topics
        if self.verbose == True:
            pprint(self.lda_model.print_topics())

    def score(self):
        # Compute Perplexity, a measure of how good the model is. Lower is better.
        print('\nPerplexity: ', self.lda_model.log_perplexity(self.tdf))

        # Compute Coherence Score, a different metric that better aligns with human
        # comprehension.
        coherence_model_lda = CoherenceModel(model=self.lda_model, 
                                             texts=self.corpus, 
                                             dictionary=self.id2word, 
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

def load_data_clean_text(file_path):
    df = pd.read_csv(file_path)

    # remove single quotes, double quotes, punctuation, and convert to lower case
    df['extracted_text'] = df['extracted_text'].str.replace('"','')
    df['extracted_text'] = df['extracted_text'].str.replace("'",'')
    df['extracted_text'] = df['extracted_text'].str.replace('[^\w\s]', '')
    df['extracted_text'] = df['extracted_text'].str.lower()

    # Create individual words from sentences by splitting on spaces
    df['extracted_text'] = df['extracted_text'].str.split()
    
    corpus_of_words = list(df['extracted_text'].values)

    return corpus_of_words

def main():
    file_path = 'data/Harry_Potter_Clean.csv'
    corpus = load_data_clean_text(file_path)
    gen_LDA = GensimTopicModeler(corpus, verbose=True)
    gen_LDA.tokenize_corpus()
    gen_LDA.fit(10)

if __name__ == "__main__":
    main()