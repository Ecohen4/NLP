import spacy
import pdb

def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


class DocumentCleaner():
    ''' perform word-level cleaning, stemming, and lemmatization using spaCy '''
    def __init__(self, doc, punctuation, lowercase=True, lemmatize=True,
                        rm_pronouns=True, rm_numbers=False, rm_terms=None):
        self.doc = doc # a spacy document
        self.punc = punctuation # list of strings
        self.do_lower = lowercase
        self.do_lemma = lemmatize
        self.rm_pronouns = rm_pronouns
        self.rm_numbers = rm_numbers
        self.rm_terms = rm_terms
        self.words = []

    def tokenize(self):
        self.tokens = [x for x in self.doc]

    def filter_tokens(self):
        if self.do_lemma:
            self.tokens = [x.lemma_ for x in self.tokens if not x.is_stop]
        else:
            self.tokens = [x.text for x in self.tokens if not x.is_stop]

    def remove_punc(self):
        out = list(set(self.tokens) - set(self.punc))
        self.tokens = out

    def remove_numbers(self):
        out = [x for x in self.tokens if not is_int(x) or not is_float(x)]
        self.tokens = out

    def lowercase(self):
        out = [x.lower() for x in self.tokens]
        self.tokens = out

    def remove_terms(self):
        out = list(set(self.tokens) - set(self.rm_terms))
        self.tokens = out

    def clean(self):
        self.tokenize()
        self.filter_tokens()
        self.remove_punc()

        if self.rm_terms is not None:
            self.remove_terms()

        if self.do_lower:
            self.lowercase()

        if self.rm_numbers:
            self.remove_numbers()

        return self.tokens

# easier
# def keep_token(t):
#     return (t.is_alpha and
#             not (t.is_space or t.is_punct or
#                  t.is_stop or t.like_num))
#
# def lemmatize_doc(doc):
#     return [ t.lemma_ for t in doc if keep_token(t)]
