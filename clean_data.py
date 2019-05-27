import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, pdb
import spacy
from argparse import ArgumentParser

from LanguageTools.spacy_utils import DocumentCleaner

def clean_chars(df, col, chars):
    ''' apply to pandas df
        replaces 'chars' in df[col]
        '''
    new_ser = df[col].apply(lambda x: x.replace(chars,''))
    df[col] = new_ser
    return df

def clean_document(text, nlp, punctuation, rm_terms):
    ''' lamba function for pandas df
        apply to feature 'tokens'
        '''
    print('cleaning...')
    doc = nlp(text)
    cleaner = DocumentCleaner(doc, punctuation, lowercase=True,
            rm_terms=rm_terms, rm_numbers=True)
    cleaner.clean()
    return cleaner.tokens

def main(args):
    raw = pd.read_csv(args.csv)

    # char cleaning: chapters
    data = clean_chars(raw, 'chapter_name', '\\u2019')

    # define punctuation
    punc_chars = '.!"“#$%&\'’()*+,-/:;<=>?@[\\]^_-`{|}~© '
    punctuation = [x for x in punc_chars]
    punc_to_add = ['…','‘','—','’s','--','    ','\xa0']
    for x in punc_to_add:
        punctuation.append(x)

    # terms to remove
    rm_terms = ['fer', 'yer', 'yeh', 'ter','mr','mrs','Mr.','Mrs.','bin','pron']

    # generate cleaned tokens of each document
    # spacy nlp
    nlp = spacy.load('en')
    data_clean = data.copy()
    data_clean['tokens'] = data.extracted_text.apply(lambda x: clean_document(x, nlp, punctuation, rm_terms))

    fname = os.path.join('pkl', args.outfile)
    with open(fname, 'wb') as f:
        pickle.dump(data_clean, f)
        print(f'{fname} written')

if __name__=='__main__':
    parser = ArgumentParser('tool to prepare HP dataset for NLP')
    parser.add_argument('csv', help='csv file')
    parser.add_argument('-o', dest='outfile', help='output pkl file with .pkl')
    args = parser.parse_args()

    if not args.outfile:
        args.outfile = 'data_clean.pkl'

    main(args)
