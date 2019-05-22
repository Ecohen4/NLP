import pandas as pd
import numpy as np

import argparse

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Various summarizer options from the sumy library
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer 
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# from nltk.corpus import stopwords


class TextSummarizer():
    def __init__(self, df):
        self.df = df
        self.sumarizer_options = [EdmundsonSummarizer, KLSummarizer, LexRankSummarizer, 
                                  LsaSummarizer, LuhnSummarizer, ReductionSummarizer,
                                  SumBasicSummarizer, TextRankSummarizer]
        self.stop_words = stopwords.words('english')

    def _summarize(self, text, summarizer, num_sentences=5, bonus_words): # all_chapters=True, all_summarizers=False
        self.summarizer = summarizer
        self.num_sentences = num_sentences
        self.language = "english"

        summarizer = self.summarizer(Stemmer(self.language))
        summarizer.stop_words = get_stop_words(self.language)
        if isinstance(summarizer, EdmundsonSummarizer):
            summarizer.bonus_words = bonus_words
            summarizer.stigma_words = ['zdfgthdvndadv']
            summarizer.null_words = stop_words
    summary = summarizer(PlaintextParser(text, Tokenizer(language)).document, sentence_count)
    return summary
        self.parser = PlaintextParser(first_chap_text, Tokenizer(language))
        summarizer = summarizer(Stemmer(language))





def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['chapter_name'] = df['chapter_name'].replace("u2019", "")
        df['chapter_name'] = df['chapter_name'].str.replace("\\", "'")

        return df

    except:
        print("Fatal Error: File '{}' could not be located, or is not readable.".format(file_path))
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', default='data/Harry_Potter_Clean.csv', help='File path/name of text to summarize')
    parser.add_argument('-l', '--length', default=5, help='Number of summary sentences to return')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = load_data(args.filepath)


if __name__ == "__main__":
    main()