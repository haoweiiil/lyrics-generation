import pandas as pd
from nltk import word_tokenize
import random
import nltk
from itertools import product as iterprod
import re
from num2words import num2words
import string
# import torch
import pickle

try:
    phones_dict = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    phones_dict = nltk.corpus.cmudict.dict()
import cmudict

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

class Dataset():

    def __init__(self, filepath, subset=None):
        # data
        df = pd.read_csv(filepath)
        self.genre_list = df['Genre'].values
        self.artist_list = df['Artist'].values
        if subset is not None:
            df = df[df['Genre'].apply(lambda x: x in subset)]

        self.lyrics_list = df['Lyrics'].values
        # chars
        self.all_characters = string.printable
        self.char_vocab_size = len(self.all_characters)
        # phones
        self.uniq_phones = cmudict.symbols()
        self.uniq_phones.append('<UNK>') # deal with unrecognized words
        self.phones_dict = cmudict.dict()
        self.phones2index = {phone: i for i, phone in enumerate(self.uniq_phones)}
        self.index2phones = {i: phone for i, phone in enumerate(self.uniq_phones)}
        self.ph_size = len(self.uniq_phones)
        # artist
        self.artist_set = list(set(self.artist_list))
        self.artist_size = len(self.artist_list)
        # genre
        self.genre_set = list(set(self.genre_list))
        self.genre_size = len(self.genre_set)
        # word index
        self.word2idx = {}
        self.idx2word = {}
        self.unique_words = []
        self.word_vocab_size = -1

    def load_dict(self):
        with open('saved_data/artist_set.p', "rb") as fp:
            self.artist_set = pickle.load(fp)
        with open('saved_data/genre_set.p', 'rb') as fp:
            self.genre_set = pickle.load(fp)
        with open('saved_data/word2idx.p', 'rb') as fp:
            self.word2idx = pickle.load(fp)
        with open('saved_data/idx2word.p', 'rb') as fp:
            self.idx2word = pickle.load(fp)
        with open('saved_data/phones2index.p', 'rb') as fp:
            self.phones2index = pickle.load(fp)
        with open('saved_data/index2phones.p', 'rb') as fp:
            self.index2phones = pickle.load(fp)

        self.word_vocab_size = len(self.word2idx)
        self.artist_size = len(self.artist_set)
        self.genre_size = len(self.genre_set)

    def save_dict(self):
        with open('saved_data/artist_set.p', "wb") as fp:
            pickle.dump(self.artist_set, fp)
        with open('saved_data/genre_set.p', 'wb') as fp:
            pickle.dump(self.genre_set, fp)
        with open('saved_data/word2idx.p', 'wb') as fp:
            pickle.dump(self.word2idx, fp)
        with open('saved_data/idx2word.p', 'wb') as fp:
            pickle.dump(self.idx2word, fp)
        with open('saved_data/phones2index.p', 'wb') as fp:
            pickle.dump(self.phones2index, fp)
        with open('saved_data/index2phones.p', 'wb') as fp:
            pickle.dump(self.index2phones, fp)


    def tokenize_corpus(self, tokenizer):
        ''' Takes entire corpus as a string and return word to index, and list of words
            :returns word2idx, unique_words, word_vocab_len'''
        ## handle lyrics
        # TODO: handle punctuation in word embedding
        lyrics = '\n'.join(self.lyrics_list)
        lyrics = re.sub('([,.!?"\n])', r' \1 ', lyrics) # pad punctuation and new line for splitting
        corp_lower = lyrics.lower()
        tokens = re.split("[ ]+", corp_lower)
        clean_tokens = [input_format_check(t) for t in tokens]
        # word level
        self.unique_words = sorted(list(set(clean_tokens)))
        self.unique_words.append('<UNK>')
        if "\n" not in self.unique_words:
            self.unique_words.append("\n")
        self.word2idx = {w: i for i, w in enumerate(self.unique_words)}  # reserve one space for padding
        self.idx2word = {i: w for i, w in enumerate(self.unique_words)}
        self.word_vocab_size = len(self.word2idx)

        return self.word_vocab_size

    def get_artist_genre(self, seq_len, artist, genre):
        ''' returns (artist_list, genre_list), each is list of length seq_len '''
        assert artist in self.artist_set
        assert genre in self.genre_set

        artist_idx = self.artist_set.index(artist)
        genre_idx = self.genre_set.index(genre)
        artist_list = [artist_idx]*seq_len
        genre_list = [genre_idx]*seq_len
        return artist_list, genre_list

    def lyric_to_idx(self, line, if_train = True):
        """
            input:
                line: one line of lyric
            output:
                word_idx_list: list, sent_len
                * only returns the following if_train == True
                char_idx_list: list, (sent_len, diff_word_len)
                ph_idx_List: list, (sent_len, diff_ph_len)
        """
        ## word_token_list: (batch_size, sent_len)
        line = line.strip(" ")
        word_token_list = re.split("[ ]+", line)

        word_token = [input_format_check(t) for t in word_token_list]
        word_idx = []
        char_idx = []
        ph_idx = []

        if not if_train:
            for word in word_token:
                if word.lower() in self.word2idx:
                    word_idx.append(self.word2idx[word.lower()])
                else:
                    word_idx.append(self.word_vocab_size - 1)  # last index reserved for <UNK>
            return word_idx

        for word in word_token:
            if word == '':
                print("line that creates empty word: ", line)
                print("word token from line:" ,word_token)
            if word.lower() in self.word2idx:
                word_idx.append(self.word2idx[word.lower()])
            else:
                word_idx.append(self.word_vocab_size - 1)  # last index reserved for <UNK>
            ## char idx
            try:
                char_idx.append([self.all_characters.index(c) for c in word])
            except:
                print("char not found in printable characters: ", word)
            ## phone idx
            ph_list = wordbreak(word)
            # if multiple phones interpretation, take the first one
            if isinstance(ph_list[0], list):
                ph_list = ph_list[0]
            ph_temp_list = [self.phones2index[ph] for ph in ph_list]
            ph_idx.append(ph_temp_list)

        return word_idx, char_idx, ph_idx

    def random_song(self, path="data/csv/train.csv", subset=["R&B"], num_lines = 2, if_train = True):
        ''':returns input_lines, target_lines, artist, genre'''
        df = pd.read_csv(path)
        if subset is not None:
            df = df[df['Genre'].apply(lambda x: x in subset)]

        curr_lyrics_list = df['Lyrics'].values
        curr_artist_list = df['Artist'].values
        curr_genre_list = df['Genre'].values

        # sample a random song
        while True:
            rand_song = random.randint(0, len(curr_lyrics_list) - 1)
            lyrics = curr_lyrics_list[rand_song]
            lyric_lines = lyrics.split("\n")
            artist = curr_artist_list[rand_song]
            genre = curr_genre_list[rand_song]
            mod_lyric_lines = []
            # remove empty lines
            for line in lyric_lines:
                if line != '':
                    mod_lyric_lines.append(line)
            # find a song with enough lines
            if len(mod_lyric_lines) > num_lines:
                break

        selected_str = re.sub('([,.\-!?()\n])', r' \1 ', '\n'.join(mod_lyric_lines))
        selected_str = selected_str.strip()
        select_word_tokens = re.split("[ ]+", selected_str)
        input_lines = ' '.join(select_word_tokens[:-1])
        target_lines = ' '.join(select_word_tokens[1:])

        return input_lines, target_lines, artist, genre

    def random_lyric_chunks(self, path="data/csv/train.csv", subset=["R&B"], num_lines = 2, if_train = True):
        ''' randomly select chunks of lines
            path: allows training and test file
            if_train == True, then input and target are off by one character,
            if_train == False, target is the last line in generation
            returns: input_line, target_line, artist, genre'''
        df = pd.read_csv(path)
        if subset is not None:
            df = df[df['Genre'].apply(lambda x: x in subset)]

        curr_lyrics_list = df['Lyrics'].values
        curr_artist_list = df['Artist'].values
        curr_genre_list = df['Genre'].values
        next_line = ''

        # if in valuation, add one more line as reference for next line prediction
        if not if_train:
            num_lines += 1

        # sample a random song
        while True:
            rand_song = random.randint(0, len(curr_lyrics_list)-1)
            lyrics = curr_lyrics_list[rand_song]
            lyric_lines = lyrics.split("\n")
            artist = curr_artist_list[rand_song]
            genre = curr_genre_list[rand_song]
            mod_lyric_lines = []
            # remove empty lines
            for line in lyric_lines:
                if line != '':
                    mod_lyric_lines.append(line)
            # find a song with enough lines
            if len(mod_lyric_lines) > num_lines:
                break

        # sample lines from song
        while True:
            lyric_idx = random.randint(0,len(mod_lyric_lines)-num_lines-1)
            selected_lines = mod_lyric_lines[lyric_idx:(lyric_idx+num_lines)]
            # if in evaluation
            if not if_train:
                next_line = selected_lines[-1]  # reference for next line prediction
                selected_lines = selected_lines[:-1]
            selected_str = '\n'.join(selected_lines)
            # strip leading and trailing new line character
            selected_str = re.sub('([,.\-!?()\n])', r' \1 ', selected_str)
            next_line = re.sub('([,.\-!?()\n])', r' \1 ', next_line)
            selected_str = selected_str.strip()
            word_tokens = re.split("[ ]+", selected_str)
            input_chunk = word_tokens[:-1]
            target_chunk = word_tokens[1:]
            # if in evaluation, use the complete input sequence
            if not if_train:
                # input line of evaluation should end with new line character
                word_tokens.append('\n')
                input_chunk = word_tokens
                target_chunk = word_tokens
            if len(word_tokens) > 2:
                break

        # print(selected_str)
        # print("input: ", input_chunk)
        # print("target: ", target_chunk)
        input_line = ' '.join(input_chunk)
        target_line = ' '.join(target_chunk)

        assert input_line != ''
        assert target_line != ''

        return input_line, target_line, artist, genre, next_line

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()


if __name__ == "__main__":
    ds = Dataset('./data/csv/train.csv', subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    # print(ds.unique_words[-1])
    # print(vocab_size)
    # wt, sl, ct, wl = ds.lyric_to_tensor("I will go to dinner")
    # print(ds.lyric_to_phones("I'm going to dinner"))
    # word, char, ph = ds.lyric_to_idx("I am leaving")
    # input_list = [[word], [char], [ph]]
    # features = [ds.get_artist_genre(3, 'kem', 'R&B')]
    # target = [ds.lyric_to_idx("am leaving here", False)]
    # input_list.append(features)
    # input_list.append(target)
    # result = ds.batchify_sequence_labeling(input_list)
    inp, target, artist, genre, next_line = ds.random_lyric_chunks(path = "data/csv/train.csv", subset=["R&B"], num_lines= 2, if_train=False)
    print(inp)
    print(target)
    print(next_line)
    print("index of new line: ", ds.word2idx['\n'])
    print(ds.lyric_to_idx(inp, False))
    # print(ds.genre_set)

