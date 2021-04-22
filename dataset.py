import nltk
from itertools import product as iterprod
import re
from num2words import num2words
import string
import torch
import numpy as np
import pandas as pd
from features import *
from nltk import word_tokenize
import random
# nltk.download('cmudict')
import cmudict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath, subset=None):
        # data
        df = pd.read_csv(filepath)
        if subset is not None:
            df = df[df['Genre'].apply(lambda x: x in subset)]

        self.lyrics_list = df['Lyrics'].values
        self.artist_list = df['Artist'].values
        self.genre_list = df['Genre'].values
        # chars
        self.all_characters = string.printable
        self.n_character = len(self.all_characters)
        # phones
        self.uniq_phones = cmudict.symbols()
        self.phones_dict = cmudict.dict()
        self.phones2index = {phone: i+1 for i, phone in enumerate(self.uniq_phones)}
        self.index2phones = {i+1: phone for i, phone in enumerate(self.uniq_phones)}
        # artist
        self.artist_set = list(set(self.artist_list))
        # genre
        self.genre_set = list(set(self.genre_list))
        # word index
        self.word2index = {}
        self.uniq_words = []
        self.word_vocab_size = -1

    def tokenize_corpus(self, tokenizer):
        ''' Takes entire corpus as a string and return word to index, and list of words
            :returns word2idx, unique_words, word_vocab_len'''
        ## handle lyrics
        lyrics = '\n'.join(self.lyrics_list)
        corp_lower = lyrics.lower()
        tokens = tokenizer(corp_lower)
        clean_tokens = [input_format_check(t) for t in tokens]
        # word level
        self.unique_words = list(set(clean_tokens))
        self.word2idx = {w: i + 1 for i, w in enumerate(self.unique_words)}
        self.unique_words.append('<UNK>')
        self.word_vocab_size = len(self.unique_words)

        return self.word_vocab_size

    def get_artist(self, seq_len, artist):
        return [artist] * seq_len

    def lyric_to_tensor(self, s):
        """
        :param s: string, lyrics
        :return: word_tensor, (
        """
        word_tokens = s.split()
        word_tokens = [input_format_check(t) for t in word_tokens]
        word_idx_list = []
        for word in word_tokens:
            if word.lower() in self.word2idx:
                word_idx_list.append(self.word2idx[word.lower()])
            else:
                word_idx_list.append(self.word_vocab_size-1) # last index reserved for <UNK>
        word_tensor = torch.Tensor(word_idx_list).contiguous().view(-1,1)
        sent_len = len(word_tokens)

        # convert every word to char representation
        char_idx = []
        word_len = []
        for word in word_tokens:
            char_list = [all_characters.index(char) for char in word]
            word_len.append(len(word))
            char_idx.append(torch.FloatTensor(char_list))
        char_tensor = torch.nn.utils.rnn.pad_sequence(char_idx, batch_first=True)
        word_len = np.array(word_len)

        return word_tensor, sent_len, char_tensor, word_len

    def lyric_to_phones(self, s):
        '''returns:
                padded tensor, (sent_len, phones_list_len)
                seq_len, num of phones in each word, np array of dim (word, 1)'''
        word_tokens = s.split()
        word_tokens = [input_format_check(t) for t in word_tokens]

        phone_list = []
        seq_len = []
        for word in word_tokens:
            ph_list = wordbreak(word)
            # if word is split into multiple tokens by cmudict
            if len(np.array(ph_list).shape) > 1:
                ph_list = sum(ph_list, [])
            try:
                ph_idx_list = [self.phones2index[ph] for ph in ph_list]
                phone_list.append(torch.Tensor(ph_idx_list))
                seq_len.append(len(ph_idx_list))
            except:
                print("Phones not found in CMU dictionary")

        phone_tensor = torch.nn.utils.rnn.pad_sequence(phone_list, batch_first=True)

        return phone_tensor, np.array(seq_len)

    def random_training_chunks(self, num_lines = 2):
        ''' randomly select chunks of lines
            returns: input_line, target_line'''
        rand_idx = random.randint(0,len(self.lyrics_list)-num_lines-1)
        selected_lines = self.lyrics_list[rand_idx:rand_idx+num_lines]
        selected_str = '\n'.join(selected_lines)
        word_tokens = word_tokenize(selected_str)
        input_chunk = word_tokens[:-1]
        target_chunk = word_tokens[1:]

        input_line = ' '.join(input_chunk)
        target_line = ' '.join(target_chunk)

        return input_line, target_line

    # Note: this function handles batch_size > 1
    def batchify_sequence_labeling(self, input_batch_list, gpu, if_train=True):
        """
            input: list of words, chars and labels, various length. [[words, chars, phones, features, target],[words, chars, phones, features, target],...]
                words: word ids for one sentence. (batch_size, sent_len)
                chars: char ids for one sentences, various length. (batch_size, sent_len, each_word_length)
                phones: phone ids one sentences, various length. (batch_size, sent_len, num_phone_in_word)
                features: genre and artist. (batch_size, sent_len, feature_nums)
                target: next word prediction. (batch_size, sent_len)
            output:
                zero padding for word and char, with their batch length
                word_seq_tensor: (batch_size, max_sent_len) Variable
                feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
                word_seq_lengths: (batch_size,1) Tensor
                char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
                char_seq_lengths: (batch_size*max_sent_len,1) Tensor
                char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
                phone_seq_tensor: (batch_size*max_sent_len, max_ph_len) Variable
                phone_seq_lengths: (batch_size*max_sent_len,1) Tensor
                phone_seq_recover: (batch_size*max_sent_len,1)  recover phone sequence order
                target_seq_tensor: (batch_size, max_sent_len)
                mask: (batch_size, max_sent_len)
        """
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        chars = [sent[1] for sent in input_batch_list]
        phones = [sent[2] for sent in input_batch_list]
        features = [np.asarray(sent[3]) for sent in input_batch_list]
        feature_num = len(features[0][0])
        # genres = [sent[2] for sent in input_batch_list]
        # artist = [sent[3] for sent in input_batch_list]
        target = [sent[4] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        target_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensors = []
        for idx in range(feature_num):
            feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
        for idx, (seq, target_word, seqlen) in enumerate(zip(words, target, word_seq_lengths)):
            seqlen = seqlen.item()
            # fill first seq_len entries with each word_seq and target_seq
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            target_seq_tensor[idx, :seqlen] = torch.LongTensor(target_word)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        target_seq_tensor = target_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]
        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]

        ### deal with phones
        # pad_phones (batch_size, max_seq_len)
        pad_phones = [phones[idx] + [[0]] * (max_seq_len - len(phones[idx])) for idx in range(len(phones))]
        length_list = [list(map(len, pad_phone)) for pad_phone in pad_phones]
        max_ph_len = max(map(max, length_list))
        phone_seq_tensor = torch.zeros((batch_size, max_seq_len, max_ph_len), requires_grad=if_train).long()
        phone_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_phones, phone_seq_tensor)):
            for idy, (ph, phlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :phlen] = torch.LongTensor(ph)
        phone_seq_tensor = phone_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
        phone_seq_lengths = phone_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
        phone_seq_lengths, ph_perm_idx = phone_seq_lengths.sort(0, descending=True)
        phone_seq_tensor = phone_seq_tensor[ph_perm_idx]

        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        _, phone_seq_recover = ph_perm_idx.sort(0, descending=False)
        # if gpu:
        #     word_seq_tensor = word_seq_tensor.cuda()
        #     for idx in range(feature_num):
        #         feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        #     word_seq_lengths = word_seq_lengths.cuda()
        #     word_seq_recover = word_seq_recover.cuda()
        #     label_seq_tensor = label_seq_tensor.cuda()
        #     char_seq_tensor = char_seq_tensor.cuda()
        #     char_seq_recover = char_seq_recover.cuda()
        #     mask = mask.cuda()
        return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
               char_seq_recover, phone_seq_tensor, phone_seq_lengths, phone_seq_recover, target_seq_tensor, mask


if __name__ == "__main__":
    ds = Dataset('./data/csv/train.csv', subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    # print(vocab_size)
    # wt, sl, ct, wl = ds.lyric_to_tensor("I will go to dinner")
    # print(ds.lyric_to_phones("I'm going to dinner"))
    input_list = [[]]
    tensors = ds.batchify_sequence_labeling()