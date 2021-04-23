import nltk
from itertools import product as iterprod
import re
from num2words import num2words
import string
import torch
import numpy as np
# download cmudict
try:
    phones_dict = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    phones_dict = nltk.corpus.cmudict.dict()
import cmudict

all_characters = string.printable
n_character = len(all_characters)
uniq_phones = cmudict.symbols()

artist_list = []
genre_list = []

# https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
# https://en.wikipedia.org/wiki/ARPABET
def wordbreak(s):
    ''' returns a list of list pronouncing according to cmudict, applies partitioning for words not in dictionary'''
    s = s.lower()
    # if doesn't contain alphabet, return unknown
    # unknown phone handled in dictionary
    if not re.search("[a-zA-Z]", s):
        return [['<UNK>']]
    # strip non-alphabetic character
    s = re.sub("[^a-z]", "", s)
    if s in phones_dict:
        return phones_dict[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in phones_dict and wordbreak(suf) is not None:
            return [x + y for x, y in iterprod(phones_dict[pre], wordbreak(suf))]
    # return empty list if cannot be converted to phones
    return [[]]

# https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones
def get_phones(s):
    ''' takes a word and returns a tensor of its phones index in cmudict '''
    ph_list = wordbreak(s)[0]
    ph_list = sum(ph_list, [])
    tensor = torch.zeros(len(ph_list)).long()
    for i in range(len(ph_list)):
        tensor[i] = uniq_phones.index(ph_list[i])
    return tensor

def get_artist(seq_len, artist):
    return [artist]*seq_len

def lyric_to_tensor(word2idx, s):
    ''' returns word-level and char-level tensor '''
    word_tokens = s.split()
    word_tokens = [input_format_check(t) for t in word_tokens]
    word_idx_list = []
    for word in word_tokens:
        if word.lower() in word2idx:
            word_idx_list.append(word2idx[word.lower()])
        else:
            word_idx_list.append()

    # convert every word to char representation
    char_idx = []
    word_len = []
    for word in word_tokens:
        char_list = [all_characters.index(char) for char in word]
        word_len.append(len(word))
        char_idx.append(torch.FloatTensor(char_list))
    char_tensor = torch.nn.utils.rnn.pad_sequence(char_idx, batch_first=True)
    word_len = np.array(word_len)


def lyric_to_phones(s):
    '''returns padded tensor, (sent_len, phones_list_len)'''
    word_tokens = s.split()
    word_tokens = [input_format_check(t) for t in word_tokens]

    phone_list = [get_phones(word) for word in word_tokens]

    phone_tensor = torch.nn.utils.rnn.pad_sequence(phone_list, batch_first=True)

    return phone_tensor

def input_format_check(s):
    '''
    takes in a word token,
    returns: empty string if it's punctuation, convert number to text,
    strip punctuation other than apostrophe  '''
    if s.isnumeric():
        return num2words(s)
    elif s.isalpha():
        return s
    elif re.search("(^')|('$)", s) and len(s) > 1:
        return re.sub("(^')|('$)", "", s)
    else:
        # print("Manually check input type: ", s)
        return s



if __name__ == "__main__":
    # print(lyric_to_phones('I will go'))
    ph_list = wordbreak("givin'")
    # if len(np.array(ph_list).shape) > 1:
    #     ph_list = ph_list[0]
    # if isinstance(ph_list[0], list):
    #     ph_list = ph_list[0]
    print(ph_list)