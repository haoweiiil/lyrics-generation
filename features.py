import nltk
from itertools import product as iterprod
import re

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



if __name__ == "__main__":
    print(get_pronounce('seaya'))