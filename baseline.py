import math, random
import numpy as np
import re
import pandas as pd

################################################################################
# Helper functions
################################################################################


def create_ngram_model(model_class, path, n=2, k=0, type='csv', genre=None):
    ''' Creates and returns a new n-gram model trained on path file '''
    """
    path: numpy file of dimension nx3, columns: artist, genre, lyrics
    
    return: updated model of model_class
    """
    num_iterations = 500  # number of samples to draw
    length = 10  # how many lines in one sample

    model = model_class(n, k)
    if type == 'txt':
        with open(path, encoding='utf-8', errors='ignore') as f:
            lyrics = f.read()
    else:
        lyrics = csv_to_text(path, genre)

    for i in range(num_iterations):
        temp = lyrics_chunks(lyrics, length)
        model.update(temp)

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

if __name__ == '__main__':
    m = create_ngram_model(NgramModel, 'data/csv/train.csv', 7, 0.0000001)
    generated_lyrics = m.random_lines(10)
    print(generated_lyrics)
