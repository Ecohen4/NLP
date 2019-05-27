import pandas as pd
import numpy as np
import sys, os, pickle, pdb
import spacy

from PotterUniverse.universe import load_char_list, load_place_names

def write_summary(sorted_pos_dict):
    '''
    algorithmically write a summary and title
    indices: top n to select (or last n in case of w5)
    approx index: ceil() of half the terms in each list

    param
    -----
    sorted_pos_dict     dict of part-of-speech keywords sorted by tfidf score
    '''

    lengths = [len(x) for x in sorted_pos_dict.values()]
    ii = [int(np.ceil(x/2)) for x in lengths]
    # special rules:
    if ii[0] >= lengths[0]/2: # remove 1 from main chars vector if it will include center name in both
        ii[0] -= 1
    if lengths[3] <= 2 and ii[3] < 2: # some places being cut out
        ii[3] = 2
    if ii[1] > 5: # limit nouns to 5
        ii[1] = 5

    # selections from ranked keywords by part-of-speech
    w1 = ['Harry'] + sorted_pos_dict['nouns_p'][:ii[0]+1] # main chars, +1 for Harry
    w2 = sorted_pos_dict['verbs'][:ii[2]]
    w3 = sorted_pos_dict['nouns'][:ii[1]]
    w4 = sorted_pos_dict['places'][:ii[3]]
    w5 = sorted_pos_dict['nouns_p'][- ii[0]:] # supporting chars
    # selections for title
    t1,t1b = w1[0],w1[1]
    tt = [] # grab first item if it exists
    for x in [w2, w3, w4, w5]:
        try:
            tt.append(x[0])
        except:
            tt.append('')

    summary = f'people: {str(w1)}, actions:{str(w2)}, with a {str(w3)}, at {str(w4)}, supporting: {str(w5)}'
    title = f'{t1} and {t1b} {tt[0]} with a {tt[1]} at {tt[2]} with {tt[3]}'

    return summary, title



def main(pkl_file, outname):
    # load tokenized, lemmatized data with 'tfidf_10' field
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # load character and place names
    char_names_all, char_names_lower = load_char_list('data/potter_characters.csv')
    place_names, place_names_lower = load_place_names()

    # corpus: ranked keywords from each document
    keyword_l = data.keywords.tolist()

    # create chapter summaries from ranked keywords and their parts of speach
    nlp = spacy.load('en_core_web_lg')
    titles = []
    summaries = []
    for keyword_tups in keyword_l:
        keyword_dict = {x[0]:x[1] for x in keyword_tups}
        keywords_ = keyword_dict.keys()

        # capitalize if in character names (for spaCy)
        # this is a manual list-membership approach. Tried [(x, x.label_)
        # for x in doc_.ents] but not working.
        keywords = []
        for x in keywords_:
            if x.lower() in char_names_lower:
                keywords.append(x.capitalize())
            else:
                keywords.append(x)
            if x == 'mrs': # this should have been removed upstream of char_names_lower
                keywords.remove(x)

        # use spaCy to tag parts of speech
        doc_ = nlp(' '.join([x for x in keywords]))
        tags = [tok.tag_ for tok in doc_]
        tags_ = np.array(tags)

        # parse parts-of-speech
        kk = np.array(keywords)
        nouns_p = kk[np.where(tags_=='NNP')[0]]
        verbs = kk[np.where(np.in1d(tags_,['VB', 'VBD']))[0]]
        # split place names from nouns
        nouns = kk[np.where(tags_=='NN')[0]]
        places = [x for x in list(nouns) if x in place_names_lower]
        nouns = [x for x in nouns if x not in places]
        places = [x.capitalize() for x in places]

        # dict of p-o-s keywords sorted by tfidf score
        ww_sort = {}
        lists = [nouns_p, nouns, verbs, places]
        labels = ['nouns_p','nouns','verbs','places']
        for lst, label in zip(lists, labels):
            ww_sort[label] = sorted(lst, key=lambda x: keyword_dict[x.lower()], reverse=True)

        # summary and title
        summary, title = write_summary(ww_sort)
        summaries.append(summary)
        titles.append(title)

    data['summaries'] = summaries
    data['title'] = titles

    # summary dataset to csv
    summ_df = data[['book_name','chapter','chapter_name','title','summaries']]
    summ_df.to_csv(f'output/{outname}.csv')
    print(f'output/{outname}.csv written')


if __name__=='__main__':
    ''' cmd line args:
        [1] pkl_file    pandas df, cleaned dataframe,
                        cols=['book_name','chapter','chapter_name'...'keywords']
        [2] out_csv     string, name of output csv
    '''
    # handle case if extension not added
    outname = sys.argv[2]
    if len(os.path.splitext(outname)) == 0:
        outname = outname + '.csv'

    main(sys.argv[1], outname)
