import nltk
from itertools import product as iterprod
import re
from num2words import num2words
import string

all_characters = string.printable
n_character = len(all_characters)

# download cmudict
try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

# https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
# https://en.wikipedia.org/wiki/ARPABET
def wordbreak(s):
    ''' returns a list of list pronouncing according to cmudict, applies partitioning for words not in dictionary'''
    # TODO: handle non-alphabetic words

    s = s.lower()
    assert s.isalpha()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x + y for x, y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

# https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones
def get_pronounce(s):
    ''' returns a list of phones in cmudict, total of 39 types'''
    ph_list = wordbreak(s)[0]
    return list(map(lambda x: re.sub('[^a-zA-Z]+$','',x), ph_list))

def get_artist(seq_len, artist):
    return [artist]*seq_len

def lyric_to_tensor(s):
    ''' returns word-level and char-level tensor '''
    word_tokens = s.split()
    word_tokens = [input_format_check(t) for t in word_tokens]
    # TODO: encode word vocab

    # TODO: convert every word to char representation
    char_idx = []
    for word in word_tokens:
        char_idx.append([all_characters.index(char) for char in word])


def lyric_to_phones(s):
    '''returns 2d list, (sent_len, varying length of phones)'''
    # TODO: complete

def input_format_check(s):
    '''
    takes in a word token,
    returns: empty string if it's punctuation, convert number to text,
    strip punctuation other than apostrophe  '''
    if s.isnumeric():
        return num2words(s)
    elif s.isalpha():
        return s
    elif re.search("[^a-zA-Z0-9']|(^')|('$)", s):
        return re.sub("[^a-zA-Z0-9']|(^')|('$)", "", s)
    else:
        print("Manually check input type: ", s)
        return s

if __name__ == "__main__":
    print(get_pronounce('seaya'))