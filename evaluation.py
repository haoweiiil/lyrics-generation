import numpy as np
import baseline
import random
import string
import torch.nn as nn
import torch
from torch.autograd import Variable

all_characters = string.printable

def avg_perplexity(model, test_df, model_type='ngram'):
    """
    computes perplexity of unseen text using given model

    :param model: learned model
    :param test_df:
    :param model_type: 'ngram' or 'neural', default is 'ngram'
    :return: average perplexity score from 20 songs
    """
    tot_pp = 0
    num_samp = 10

    if model_type == 'ngram':
        # select 20 random songs from test data and compute perplexity
        for i in range(num_samp):
            idx = random.randint(0,len(test_df)-1)
            lyric = test_df.iloc[idx]['Lyrics']
            tot_pp += model.perplexity(lyric)

    if model_type == 'neural':
        for i in range(num_samp):
            idx = random.randint(len(test_df))
            lyric = test_df.iloc[idx]['Lyrics']
            tot_pp += nn_perplexity(lyric, model)

    return tot_pp / num_samp

def nn_perplexity(val_text, model):
    input = char_tensor(val_text[:-1])
    target = char_tensor(val_text[1:])
    hidden = model.init_hidden()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for i in range(len(val_text)-1):
        output, hidden = model(input[i], hidden)
        loss += criterion(output, target[i].view(-1))
    avg_loss = loss.item()/(len(val_text)-1)
    return np.exp(avg_loss)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)