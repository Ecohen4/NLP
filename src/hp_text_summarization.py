import pandas as pd
import numpy as np

import argparse

# sumy for tokenizing and stemming
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# various summarizer options from the sumy library
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer 
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from nltk.corpus import stopwords

class TextSummarizer():
    '''
    Takes in a dataframe of the Harry Potter dataset, and summarizes each chapter using a 
    specified summarizer from the sumy library.
    The user can specify the input file to use, the number of summary sentences, the summarizer
    algorithm to use, and to save to a txt file or print to the terminal to use through 
    command line argument parsing.

    Parameters
    ----------
    df: Pandas Dataframe containing Harry Potter .csv information

    Outputs
    ----------
    Text file of summarizations or output printed to terminal
    '''

    def __init__(self, df):
        self.df = df
        self.summarizer_options_dict = {'EdmundsonSummarizer': EdmundsonSummarizer, 
                                       'KLSummarizer': KLSummarizer, 
                                       'LexRankSummarizer': LexRankSummarizer, 
                                       'LsaSummarizer': LsaSummarizer, 
                                       'LuhnSummarizer': LuhnSummarizer, 
                                       'ReductionSummarizer': ReductionSummarizer,
                                       'SumBasicSummarizer': SumBasicSummarizer, 
                                       'TextRankSummarizer': TextRankSummarizer}
        self.language = "english"
        self.stop_words = stopwords.words('english')

    def _summarize(self, text, summarizer, num_sentences, bonus_words=['Harry']):
        '''
        Summarizes an individual chapter within the Harry Potter corpus.
        The summary includes num_sentences total sentences.

        Parameters
        ----------
        text: text of the chapter
        summarizer: the chosen summarizer algorithm
        num_sentences: number of sentences to use for summary
        bonus_words: words to focus on if using Edmundson Summarizer

        Returns
        ----------
        summary: Summary of chapter, containing num_sentences sentences
                 Note this is a "summary" sumy object
        '''
        # Initialize summarizer model and stopwords
        summarizer = summarizer(Stemmer(self.language))
        summarizer.stop_words = get_stop_words(self.language)
        
        # Edmundson is special case that uses bonus_words, stigma words, and null_words
        if isinstance(summarizer, EdmundsonSummarizer):
            summarizer.bonus_words = bonus_words
            summarizer.stigma_words = ['zdfgthdvndadv']
            summarizer.null_words = self.stop_words
        
        summary = summarizer(PlaintextParser(text, Tokenizer(self.language)).document, num_sentences)
        
        return summary

    def _print_summary(self, summary, file_txt, save_to_txt):
        for sentence in summary:
            file_txt.write(str(sentence) + '\n') if save_to_txt == True else print(sentence)

    def get_summaries(self, summarizer=TextRankSummarizer, num_sentences=5, save_to_txt=True):
        '''
        Summarizes all chapters for each Harry Potter book, and either prints summaries to terminal,
        or saves to text file, depending on boolean variable save_to_txt
        '''
        file_txt = open("../summaries/harry_potter_summaries.txt","w") if save_to_txt == True else False 

        if save_to_txt == True:
            print('Generating Summaries and Saving to txt File...this should take a few minutes')
            file_txt.write('Summarizer Used: ' + str(summarizer) + '\n')
        else:
            print(f'Summarizer Used: {summarizer}')

        if type(summarizer) == str:
            summarizer = self.summarizer_options_dict[summarizer]

        # Loop through all books in the corpus
        unique_books = self.df.book_name.unique()
        for book_name in unique_books:
            if save_to_txt == True:
                print(f'Summarizing "{book_name}"...')
                file_txt.write('\n')
                file_txt.write('***** ' + book_name + ' *****')
                file_txt.write('\n')
                file_txt.write('\n')
            else:
                print()
                print('***** ' + book_name + ' *****')
                print()

            # Loop through all chapters in each book, calculate summary, and either save
            # to text file, or print to terminal
            unique_chapters = self.df[self.df['book_name'] == book_name].chapter.unique()
            for chapter in unique_chapters:
                chapter_title = self.df[(self.df['book_name'] == book_name) & 
                                    (self.df['chapter'] == chapter)] \
                                    .chapter_name.values[0]

                chapter_text = self.df[(self.df['book_name'] == book_name) & 
                                    (self.df['chapter'] == chapter)] \
                                    .extracted_text.values[0]

                summary = self._summarize(chapter_text,
                                          summarizer,
                                          num_sentences,
                                          bonus_words=chapter_text.split())
                
                if save_to_txt == True:
                    file_txt.write('Chapter Number: ' + str(chapter) + '\n')
                    file_txt.write('Chapter Number: ' + str(chapter_title) + '\n')
                    file_txt.write('Chapter Summary: \n')
                    self._print_summary(summary, file_txt, save_to_txt)
                    file_txt.write('\n')
                else:
                    print(f'Chapter Number: {chapter}')
                    print(f'Chapter Title: {chapter_title}')
                    print('Chapter Summary:')
                    self._print_summary(summary, file_txt, save_to_txt)
                    print()
                
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['chapter_name'] = df['chapter_name'].str.replace("u2019", "")
        df['chapter_name'] = df['chapter_name'].str.replace("\\", "'")

        return df

    except:
        print("Fatal Error: File '{}' could not be located, or is not readable.".format(file_path))
        exit()

def str2bool(v):
    '''
    Allows for the use of booleans in argument parsing.
    Otherwise, booleans are simply treated as strings.
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
    parser.add_argument('-summarizer', 
                        default='TextRankSummarizer', 
                        help='Summarizer to use for each chapter summary. \
                              Options are in self.summarizer_options_dict')
    parser.add_argument('-length', 
                        default=5, 
                        help='Number of summary sentences to return')
    parser.add_argument('-savetxt', 
                        default=True, 
                        type=str2bool, 
                        nargs='?', 
                        const=True, 
                        help='Save output to text file?')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = load_data(args.filepath)
    text_sum = TextSummarizer(df)
    text_sum.get_summaries(summarizer=args.summarizer,
                           num_sentences=args.length, 
                           save_to_txt=args.savetxt)

if __name__ == "__main__":
    main()