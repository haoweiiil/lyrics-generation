import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from features import *

# https://github.com/jiesutd/NCRFpp/blob/master/model/charbilstm.py
class CharBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout, bidirect_flag = True):
        super(CharBiLSTM, self).__init__()
        print("build char sequence feature extractor: LSTM ...")
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        # TODO: do we need character embedding?
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirect_flag)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        ## char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)

# https://github.com/jiesutd/NCRFpp/blob/master/model/wordrep.py
class wordRep(nn.Module):
    def __init__(self, spec_dict, char_hidden_dim, char_emb_dim, char_model_type, word_vocab_size, word_emb_dim,
                 pre_train_word_embedding, feature_emb_dim, use_artist, dropout):
        super(wordRep, self).__init__()
        self.char_hidden_dim = char_hidden_dim
        self.char_embedding_dim = char_emb_dim

        if char_model_type == 'LSTM':
            self.char_feature = CharBiLSTM(spec_dict['char_vocab_size'], self.char_embedding_dim,
                                           self.char_hidden_dim, dropout)
            self.ph_feature = CharBiLSTM(spec_dict['ph_size'], self.char_embedding_dim, self.char_hidden_dim, dropout)
        self.embedding_dim = word_emb_dim
        self.drop = nn.Dropout(dropout)
        self.word_embedding = nn.Embedding(word_vocab_size, self.embedding_dim)
        if pre_train_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_train_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(word_vocab_size, self.embedding_dim)))

        # artist embedding
        self.use_artist = use_artist
        if use_artist:
            self.artist_emb_dim = feature_emb_dim
            self.artist_embedding = nn.Embedding(spec_dict['artist_size'], self.artist_emb_dim)
            self.artist_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(spec_dict['artist_size'],
                                                                                           self.artist_emb_dim)))
        # genre embedding
        self.genre_emb_dim = feature_emb_dim
        self.genre_embedding = nn.Embedding(spec_dict['genre_size'], self.genre_emb_dim)
        self.genre_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(spec_dict['genre_size'],
                                                                                       self.genre_emb_dim)))
        # TODO: add other self-defined features if needed

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
        input
            :param word_inputs: (batch_size, sent_len)
            :param genre_input: (batch_size, 1)
            :param artist_input: None or (batch_size, 1)
            :param word_seq_lengths: (batch_size, 1)
            :param char_inputs: (batch_size*sent_len, word_length)
            :param char_seq_lengths: (batch_size*sent_len, 1)
            :param char_seq_recover: variable which records the char order information, used to recover char order
        output
            :return: Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        word_embs = self.word_embedding(word_inputs)

        word_list = [word_embs]
        word_list.append(self.genre_embedding(genre_input))
        if self.use_artist:
            word_list.append(self.artist_embedding(artist_input))
        char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.numpy())
        char_features = char_features[char_seq_recover]
        char_features = char_features.view(batch_size, sent_len, -1)
        # concat word and char features
        word_list.append(char_features)
        # word_embs = torch.cat([word_embs, char_features], 2)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)

        return word_represent

# https://github.com/jiesutd/NCRFpp/blob/master/model/wordsequence.py
class WordSequence(nn.Module):
    def __init__(self, spec_dict, dropout, num_lstm_layer, bilstm_flag=True):
        super(WordSequence, self).__init__()
        self.use_artist = spec_dict['use_artist']


if __name__ == "__main__":
    spec_dict = {"char_vocab_size":100,
                 "ph_size": 39,
                 "artist_size": -1,
                 "genre_size": -1,
                 "dropout": 0.5,
                 "num_lstm_layer": 2,
                 "bilstm_flag": True,
                 "use_artist": False}