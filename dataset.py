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
        self.unique_words = list(set(clean_tokens))
        self.unique_words.append('<UNK>')
        if "\n" not in self.unique_words:
            self.unique_words.append("\n")
        self.word2idx = {w: i for i, w in enumerate(self.unique_words)}  # reserve one space for padding
        self.idx2word = {i: w for i, w in enumerate(self.unique_words)}
        self.word_vocab_size = len(self.word2idx)

        return self.word_vocab_size

    def get_artist_genre(self, seq_len, artist, genre):
        assert artist in self.artist_set
        assert genre in self.genre_set

        artist_idx = self.artist_set.index(artist)
        genre_idx = self.genre_set.index(genre)
        feature = []
        for i in range(seq_len):
            feature.append([artist_idx, genre_idx])
        return feature

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

    def random_lyric_chunks(self, num_lines = 5):
        ''' randomly select chunks of lines
            if_train == True, then input and target are off by one character,
            if_train == False, target is the last line in generation
            returns: input_line, target_line, artist, genre'''

        # sample a random song
        while True:
            rand_song = random.randint(0, len(self.lyrics_list)-1)
            lyrics = self.lyrics_list[rand_song]
            lyric_lines = lyrics.split("\n")
            artist = self.artist_list[rand_song]
            genre = self.genre_list[rand_song]
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
            selected_str = '\n'.join(selected_lines)
            # strip leading and trailing new line character
            selected_str = re.sub('([,.!?()\n])', r' \1 ', selected_str)
            selected_str = selected_str.strip()
            word_tokens = re.split("[ ]+", selected_str)
            input_chunk = word_tokens[:-1]
            target_chunk = word_tokens[1:]
            if len(word_tokens) > 2:
                break

        # print(selected_str)
        # print("input: ", input_chunk)
        # print("target: ", target_chunk)
        input_line = ' '.join(input_chunk)
        target_line = ' '.join(target_chunk)

        assert input_line != ''
        assert target_line != ''

        return input_line, target_line, artist, genre


if __name__ == "__main__":
    ds = Dataset('./data/csv/train.csv', subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    print(ds.unique_words[-1])
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
    inp, target, artist, genre = ds.random_training_chunks(1)
    print(inp)
    print(target)
    print(artist)



# def lyric_to_tensor(self, s):
#     """
#     :param s: string, lyrics
#     :return: word_tensor, (
#     """
#     word_tokens = s.split()
#     word_tokens = [input_format_check(t) for t in word_tokens]
#     word_idx_list = []
#     for word in word_tokens:
#         if word.lower() in self.word2idx:
#             word_idx_list.append(self.word2idx[word.lower()])
#         else:
#             word_idx_list.append(self.word_vocab_size-1) # last index reserved for <UNK>
#     word_tensor = torch.Tensor(word_idx_list).contiguous().view(-1,1)
#     sent_len = len(word_tokens)
#
#     # convert every word to char representation
#     char_idx = []
#     word_len = []
#     for word in word_tokens:
#         char_list = [all_characters.index(char) for char in word]
#         word_len.append(len(word))
#         char_idx.append(torch.FloatTensor(char_list))
#     char_tensor = torch.nn.utils.rnn.pad_sequence(char_idx, batch_first=True)
#     word_len = np.array(word_len)
#
#     return word_tensor, sent_len, char_tensor, word_len

# def lyric_to_phones(self, s):
#     '''returns:
#             padded tensor, (sent_len, phones_list_len)
#             seq_len, num of phones in each word, np array of dim (word, 1)'''
#     word_tokens = s.split()
#     word_tokens = [input_format_check(t) for t in word_tokens]
#
#     phone_list = []
#     seq_len = []
#     for word in word_tokens:
#         ph_list = wordbreak(word)
#         # if word is split into multiple tokens by cmudict
#         if len(np.array(ph_list).shape) > 1:
#             ph_list = sum(ph_list, [])
#         try:
#             ph_idx_list = [self.phones2index[ph] for ph in ph_list]
#             phone_list.append(torch.Tensor(ph_idx_list))
#             seq_len.append(len(ph_idx_list))
#         except:
#             print("Phones not found in CMU dictionary")
#
#     phone_tensor = torch.nn.utils.rnn.pad_sequence(phone_list, batch_first=True)
#
#     return phone_tensor, np.array(seq_len)