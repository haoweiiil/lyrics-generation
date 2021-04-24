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
        # print("build char sequence feature extractor: LSTM ...")
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
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
class WordRep(nn.Module):
    def __init__(self, spec_dict, dataset):
        super(WordRep, self).__init__()
        self.char_hidden_dim = spec_dict["char_hidden_dim"]
        self.char_embedding_dim = spec_dict["char_emb_dim"]

        if spec_dict["char_model_type"] == 'LSTM':
            self.char_feature = CharBiLSTM(dataset.char_vocab_size, self.char_embedding_dim,
                                           self.char_hidden_dim, spec_dict["dropout"])
            self.ph_feature = CharBiLSTM(dataset.ph_size, self.char_embedding_dim, self.char_hidden_dim, spec_dict["dropout"])
        self.embedding_dim = spec_dict["word_emb_dim"]
        self.drop = nn.Dropout(spec_dict["dropout"])
        self.word_embedding = nn.Embedding(dataset.word_vocab_size, self.embedding_dim)
        if spec_dict["pre_train_word_embedding"] is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(spec_dict["pre_train_word_embedding"]))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(dataset.word_vocab_size, self.embedding_dim)))

        # artist embedding
        self.use_artist = spec_dict["use_artist"]
        if self.use_artist:
            self.artist_emb_dim = spec_dict["feature_emb_dim"]
            self.artist_embedding = nn.Embedding(dataset.artist_size, self.artist_emb_dim)
            self.artist_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(dataset.artist_size,
                                                                                           self.artist_emb_dim)))
        # genre embedding
        self.genre_emb_dim = spec_dict["feature_emb_dim"]
        self.genre_embedding = nn.Embedding(dataset.genre_size, self.genre_emb_dim)
        self.genre_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(dataset.genre_size,
                                                                                       self.genre_emb_dim)))
        # add other self-defined features if needed

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                ph_inputs, ph_seq_lengths, ph_seq_recover):
        """
        input
            :param word_inputs: (batch_size, sent_len)
            :param genre_input: (batch_size, 1)
            :param artist_input: None or (batch_size, 1)
            :param word_seq_lengths: (batch_size, 1)
            :param char_inputs: (batch_size*sent_len, word_length)
            :param char_seq_lengths: (batch_size*sent_len, 1)
            :param char_seq_recover: variable which records the char order information, used to recover char order
            :param ph_inputs: (batch_size*sent_len, phone_list_len)
            :param ph_seq_lengths: (batch_size*sent_len, 1)
            :param ph_seq_recover: variable which records the phone order information
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
        # char hidden
        char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.numpy())
        char_features = char_features[char_seq_recover]
        char_features = char_features.view(batch_size, sent_len, -1)
        # phones hidden
        ph_features = self.ph_feature.get_last_hiddens(ph_inputs, ph_seq_lengths.numpy())
        ph_features = ph_features[ph_seq_recover]
        ph_features = ph_features.view(batch_size, sent_len, -1)
        # concat word, char, and phones features
        word_list.append(char_features)
        word_list.append(ph_features)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)

        return word_represent

# https://github.com/jiesutd/NCRFpp/blob/master/model/wordsequence.py
class WordSequence(nn.Module):
    def __init__(self, spec_dict, dataset):
        super(WordSequence, self).__init__()
        self.droplstm = nn.Dropout(spec_dict['dropout'])
        self.use_artist = spec_dict['use_artist']
        self.bilstm_flag = spec_dict['bilstm_flag']
        self.wordrep = WordRep(spec_dict, dataset)
        self.input_size = spec_dict['word_emb_dim']
        # char emb
        self.input_size += spec_dict['char_hidden_dim']
        # phones emb
        self.input_size += spec_dict['char_hidden_dim']
        self.input_size += spec_dict['feature_emb_dim']
        self.lstm_layers = spec_dict['num_lstm_layers']
        if spec_dict['use_artist']:
            self.input_size += spec_dict['feature_emb_dim']
        if self.bilstm_flag:
            lstm_hidden = spec_dict["final_hidden_dim"] // 2
        else:
            lstm_hidden = spec_dict["final_hidden_dim"]

        # build word-level lstm
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layers, batch_first=True,
                            bidirectional=self.bilstm_flag)
        self.hidden2tag = nn.Linear(spec_dict["final_hidden_dim"], dataset.word_vocab_size)

    def forward(self, word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                ph_inputs, ph_seq_lengths, ph_seq_recover):
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
            :return: Variable(batch_size, sent_len, word_vocab_size)
        """
        word_represent = self.wordrep(word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,ph_inputs, ph_seq_lengths, ph_seq_recover)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs


if __name__ == "__main__":
    spec_dict = {"ph_size": 39,
                 "dropout": 0.5,
                 "num_lstm_layer": 2,
                 "bilstm_flag": True,
                 "use_artist": True,
                 "char_hidden_dim": 128,
                 "char_emb_dim": 50,
                 "char_model_type": "LSTM",
                 "word_emb_dim": 128,
                 "pre_train_word_embedding": None,
                 "feature_emb_dim": 128,
                 "final_hidden_dim": 512,
                 "iterations": 10}