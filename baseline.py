import math, random
import numpy as np
import re
import pandas as pd
from evaluation import *
from string import punctuation

################################################################################
# Helper functions
################################################################################


def create_ngram_model(model_class, path, n=2, k=0, type='csv', genre=None):
    ''' Creates and returns a new n-gram model trained on path file '''
    """
    path: numpy file of dimension nx3, columns: artist, genre, lyrics
    
    return: updated model of model_class
    """

    model = model_class(n, k)

    df = pd.read_csv(path)
    df = df[df['Genre']==genre]
    for i in range(len(df)):
        lyric = df.iloc[i]['Lyrics']
        model.update(lyric)

    return model

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    adjusted_txt = start_pad(n) + text
    gramlist = []
    for i in range(n, len(adjusted_txt)):
        ngram = (adjusted_txt[i-n:i], adjusted_txt[i])
        gramlist.append(ngram)
    return gramlist

def csv_to_text(path, genre=None):
    ''' Takes in a csv file and return the lyrics concatenated into one text file'''
    df = pd.read_csv(path)
    if genre:
        df = df[df['Genre']==genre]
    lyrics = '\n'.join(list(df['Lyrics']))
    return lyrics

def lyrics_chunks(lyrics, length):
    """
    randomly samples lyrics chunks of specified length

    :param lyrics: str, entire lyrics corpus
    :param length: int, number of lines in one lyrics chunks
    :return: str, lyric lines joined by new line
    """
    lyrics_lines = lyrics.split('\n')
    start_idx = random.randint(0, len(lyrics_lines)-1)
    end_idx = min(start_idx+length, len(lyrics_lines))
    selected_lyrics = lyrics_lines[start_idx:end_idx]
    return '\n'.join(selected_lyrics)


def avg_perplexity(model, test_df):
    """
    computes perplexity of unseen text using given model

    :param model: learned model
    :param test_df:
    :param model_type: 'ngram' or 'neural', default is 'ngram'
    :return: average perplexity score from 20 songs
    """
    tot_pp = 0
    num_samp = 200

    # select 20 random songs from test data and compute perplexity
    for i in range(num_samp):
        idx = random.randint(0,len(test_df)-1)
        lyric = test_df.iloc[idx]['Lyrics']
        tot_pp += model.perplexity(lyric)

    return tot_pp / num_samp

def random_lines(path, genre=None):
    ''' Returns a tuple of neighboring lyrics lines from the same song '''
    df = pd.read_csv(path)
    if genre:
        df = df[df['Genre']==genre]
    while True:
        # randomly select of song
        song_idx = random.randint(0, len(df)-1)
        # randomly select line idx
        lyrics = df.iloc[song_idx]['Lyrics']
        lyric_lines = lyrics.split('\n')
        idx = random.randint(0, len(lyric_lines)-2)
        # start over if the target sentence doesn't have any alphabet
        if re.search("[a-zA-Z]", lyric_lines[idx+1]) and re.search("[a-zA-Z]", lyric_lines[idx]):
            break

    return (lyric_lines[idx], lyric_lines[idx+1])


def repeated_evaluate(model, path,genre=None,num_iterations=1):
    ''' returns result dict, average result dict, list of target and generated texts'''
    input_text, target_text = random_lines(path, genre=genre)
    generated_text = model.gen_next_sent(input_text)
    result = run_evaluations(input_text, [target_text], generated_text)
    result_list = [result]
    texts = (input_text, target_text, generated_text)
    text_list = [texts]

    func_list = list(result.keys())
    value_dict = {}
    for func in func_list:
        value_dict[func] = np.zeros(num_iterations)
        value_dict[func][0] = result[func]*1

    for i in range(1,num_iterations):
        input_text, target_text = random_lines(path, genre=genre)
        generated_text = model.gen_next_sent(input_text)
        texts = (input_text, target_text, generated_text)
        text_list.append(texts)
        temp_res = run_evaluations(input_text, [target_text], generated_text)
        result_list.append(temp_res)
        for func in func_list:
            value_dict[func][i] = temp_res[func]

    avg_result = {}
    for func in func_list:
        avg_result[func] = np.nanmean(value_dict[func])
    return result_list, avg_result, text_list

################################################################################
# Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocabs = []
        self.context_list = {}
        self.context_count = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(''.join(self.vocabs))

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        self.vocabs.append(text)
        # self.sorted_vocabs = list(set(self.vocabs))
        # self.sorted_vocabs.sort()
        gramlist = ngrams(self.n, text)
        for pair in gramlist:
            context = pair[0]
            char = pair[1]
            if context in self.context_list: # context previously showed up
                self.context_count[context]+=1
                if char in self.context_list[context]: # combination previously showed up
                    self.context_list[context][char]+=1
                else:
                    self.context_list[context][char]=1
            else:
                self.context_count[context]=1
                self.context_list[context] = {char: 1}

    def prob(self, context, char, lambdas=None):
        ''' Returns the probability of char appearing after context '''
        if context not in self.context_list:
            return 1/len(self.get_vocab())

        # if char not in self.context_list[context]:
        #     return 0.0
        # else:
        #     return self.context_list[context][char]/self.context_count[context]
        count = 0
        if char in self.context_list[context]:
            count = self.context_list[context][char]
        p = (count+self.k)/(self.context_count[context]+self.k*len(self.get_vocab()))
        return p

    def random_char(self, context):
        ''' Returns a random character based on the given context and the
            n-grams learned by this model '''
        vocabs = list(self.get_vocab())
        vocabs.sort()
        total_prob = 0
        r = random.random()
        for i in range(len(vocabs)):
            total_prob += self.prob(context, vocabs[i])
            if total_prob > r:
                return vocabs[i]
        return vocabs[-1]

    def random_text(self, length, context=None):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        if context == None:
            context = start_pad(self.n)
        else:
            context = context[-self.n:]
        text = ''
        for i in range(length):
            char = self.random_char(context)
            text += char
            context = context[1:]+char
        return text

    def random_lines(self, num_lines, context=None):
        ''' Returns specified number of lines based on learned n-grams model'''
        if context == None:
            context = start_pad(self.n)
        else:
            context = context[-self.n:]
        text = ''
        count = 0
        while True:
            char = self.random_char(context)
            if char == '\n':
                count += 1
                if count == num_lines:
                    break
            text += char
            context = context[1:] + char
        return text

    def perplexity(self, text,lambdas=None):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        padded_text = start_pad(self.n) + text
        context = start_pad(self.n)
        log_pp = 0
        for i in range(self.n, len(padded_text)):
            p = self.prob(context, padded_text[i])
            if p == 0:
                return float('inf')
            log_pp += math.log(p)
            context = context[1:]+padded_text[i]
        log_pp = log_pp/(-len(text))
        pp = math.exp(log_pp)
        return pp

    def gen_next_sent(self, input_sent):
        ''' Returns the next sentence prediction based on input sentence '''
        input_sent = start_pad(self.n) + input_sent + '\n'
        context = input_sent[-self.n:]
        next_sent = ''
        while True:
            char = self.random_char(context)
            if char != '\n':
                next_sent += char
                context = context[1:] + char
            # if hit new line character but no alphabet in sentence, start over
            elif not re.search("[a-zA-Z]", next_sent):
                context = input_sent[-self.n:]
                next_sent = ''
            else:
                break
        return next_sent


if __name__ == '__main__':
    m = create_ngram_model(NgramModel, 'data/csv/train.csv', 6, 0.0000001,genre='R&B')
    # input_text, target_text = random_lines('data/csv/train.csv', genre='R&B')
    # print("input text: " + input_text)
    # print("target text: " + target_text)
    # generated_text = m.gen_next_sent(input_text)
    # print("predicted text: " + generated_text)
    # print(evaluate_score(target_text, generated_text))
    result, avg_result, text_list = repeated_evaluate(m, 'data/csv/test.csv', 'R&B', 20)
    print(avg_result)
    # print(train_result)
    # print(text_list)


