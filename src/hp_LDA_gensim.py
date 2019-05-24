
import pandas as pd
import numpy as np

# Spacy used primarily for lemmatization
import spacy

# Gensim for Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# Import stopwords from NLTK
from nltk.corpus import stopwords

from pprint import pprint
import argparse

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

class GensimTopicModeler():
    '''
    Tokenizes corpus by removing stopwords, creating bigrams, and lemmatizing words.
    Uses the Gensim library's application of Latent Dirichlet Allocation to model 
    topics within the Harry Potter corpus.

    The user can specify the input file to use, number of topics, verbose mode,
    and weather or not to save the LDA model, through argument parsing.

    Parameters:
    ----------
    Corpus: List of lists, with each sublist containing all words in each chapter.

    Returns: 
    ----------
    See 'fit', and 'compute_coherence_values' methods below.
    '''

    def __init__(self, corpus, verbose, save_model):
        self.corpus = corpus
        self.verbose = verbose
        self.save_model = save_model

    def _remove_stopwords(self, corpus):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['not', 'look', 'do', 'go', 'get', 'would', 'be', 's', 'say',
                           'see', 'could', 'back', 'know', 'come', 'harry', 'hermione',
                           'think', 'tell', 'take', 'make', 'want'])

        return [[word for word in doc if word not in self.stop_words] for doc in corpus]
        
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
        '''
        Remove stopwords from corpus, builds bigrams, and lemmatizes corpus.
        '''

        self.corpus = self._remove_stopwords(self.corpus)
        self.corpus = self._create_bigrams(self.corpus)
        self.corpus = self._lemmatization(self.corpus)

        # Remove stopwords that snuck through after lemmatizing
        self.corpus = self._remove_stopwords(self.corpus)

    def fit(self, num_topics):
        '''
        Create dictionary of tokens, tf matrix, then fit LDA model.

        Parameters
        ----------
        num_topics: Number of topics to use in the LDA model

        Outputs
        ----------
        Printed keywords for each topic if 'verbose' == True
        Saves Gensim LDA model if 'savemodel' == True
        '''

        self.id2word = corpora.Dictionary(self.corpus)
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

        # Print the top 10 keywords for each topic
        if self.verbose:
            print("Top 10 Most Important Words by Topic:")
            pprint(self.lda_model.print_topics())

        if self.save_model:
            self.lda_model.save("../lda_model/hp_lda_model")

    def score(self):
        '''
        Compute Perplexity, a measure of how good the model is. Lower is better.
        Also computes Coherence Score, a different metric that better aligns with human
        comprehension.
        '''

        print('\nPerplexity: ', self.lda_model.log_perplexity(self.tdf))

        coherence_model_lda = CoherenceModel(model=self.lda_model, 
                                             texts=self.corpus, 
                                             dictionary=self.id2word, 
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def get_topic_distribution(self, chapter_index):
        '''
        Print the topic distribution of a single chapter, using the chapter index.
        '''
        print(f'\nTopic Distribution for Chapter at Index {chapter_index}:')
        print('format is (topic #, % of chapter comprised by topic)')
        print(self.lda_model.get_document_topics(self.tdf[chapter_index]))

    def compute_coherence_values(self, limit=20, start=2, step=2):
        """
        Compute c_v coherence for various number of topics
        Note this takes a very long time to run, but is necessary if you want
        to plot coherence vs. num_topics.

        Parameters:
        ----------
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with 
        respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            lda_model = gensim.models.ldamodel.LdaModel(corpus=self.tdf,
                                            id2word=self.id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
            model_list.append(lda_model)
            coherence_model_lda = CoherenceModel(model=lda_model, 
                                                 texts=self.corpus, 
                                                 dictionary=self.id2word, 
                                                 coherence='c_v')
            coherence_values.append(coherence_model_lda.get_coherence())

        return model_list, coherence_values

    def plot_coherence(self, coherence_values):
        '''
        Plot coherence vs num_topics. Note that the 'compute_coherence_values' method
        must be run prior to this.
        '''
        fig = plt.figure(figsize=(12,6))
        limit=20; start=2; step=2;
        x = range(start, limit, step)
        
        plt.plot(x, coherence_values, label="Coherence Score", color='g')
        
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Coherence Score vs. Number of Topics - Gensim LDA Model")
        
        plt.show()

    def plot_wordclouds(self, num_topics):
        '''
        Plot Wordcloud of Top N words in each topic.
        Note this plot works best within a jupyter notebook
        '''
        
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        cloud = WordCloud(stopwords=self.stop_words,
                        background_color='white',
                        width=2500,
                        height=1800,
                        max_words=10,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: cols[i],
                        prefer_horizontal=1.0)

        topics = self.lda_model.show_topics(formatted=False)

        fig, axes = plt.subplots(3, 3, figsize=(16,16), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            if i > num_topics - 1:
                break
            else:
                topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        plt.show()

def load_data_clean_text(file_path):
    try:
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
    except:
        print("Fatal Error: File '{}' could not be located, or is not readable.".format(file_path))
        exit()

def str2bool(v):
    '''
    Allows for the use of booleans in argument parsing.
    Otherwise, booleans are simply treated as strings
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', 
                        default='../data/Harry_Potter_Clean.csv', 
                        help='File path/name of text to summarize')
    parser.add_argument('-numtopics',
                        type=int, 
                        default=8, 
                        help='Number topics to use in LDA model')
    parser.add_argument('-verbose', 
                        default=True, 
                        type=str2bool, 
                        nargs='?', 
                        const=True, 
                        help='Print topic keywords to terminal?')
    parser.add_argument('-savemodel', 
                        default=True, 
                        type=str2bool, 
                        nargs='?', 
                        const=True, 
                        help='Save outputs of LDA model?')

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    corpus = load_data_clean_text(args.filepath)
    gen_LDA = GensimTopicModeler(corpus, args.verbose, args.savemodel)
    gen_LDA.tokenize_corpus()
    gen_LDA.fit(args.numtopics)
    gen_LDA.score()
    gen_LDA.get_topic_distribution(0)

    # uncomment if would like to plot coherence score vs. num_topics
    # Note this will take a long time to run, since creates many LDA models.
    # model_list, coherence_values = gen_LDA.compute_coherence_values()
    # gen_LDA.plot_coherence(coherence_values)

    # uncomment if would like to plot wordcloud
    # Note this plot looks best within a jupyter notebook
    # gen_LDA.plot_wordclouds(args.numtopics)

if __name__ == "__main__":
    main()