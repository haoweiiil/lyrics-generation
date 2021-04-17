import click
import numpy as np
import sys
import language_tool_python
import pronouncing
from espeakng import ESpeakNG
from Phyme import Phyme
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from bert_score import score
from bert_score import plot_example
from lexical_diversity import lex_div as ld
import re
from num2words import num2words
import requests, json

# Initiate.

tool_en = language_tool_python.LanguageTool("en-US")
ph = Phyme()
rouge = Rouge()

word_phone_url = "https://raw.githubusercontent.com/jameswenzel/Phyme/master/Phyme/data/word_phone.json"
resp = requests.get(word_phone_url)
word_phone_dict = json.loads(resp.text)

# Grammar.

def get_grammar_score(text, tool, output_error_matches=False):
    """
    check grammar using language_tool_python package
    :param text: the text that needs a grammar check, a string
    :param tool: the language specific tool to perform grammar check
    :param output_error_matches: boolean - if output the actual error or not; default is False
    :return: number of error matches / length of the text string on the character level
    """
    matches = tool.check(text)
    if output_error_matches:
        for match in matches:
            print(match.ruleId, match.replacements)
    return len(matches) / len(text)


# Rhymes.

def check_rhyme_pronouncing(input_text, generated_text, print_list=False):
    """
    a stricter version of checking rhyme using package pronouncing (CMU pronouncing)
    :param print_list:
    :param input_text: input text of the lyrics, could be one or two lines
    :param generated_text: output text of the lyrics, one line
    :return: a boolean indicate if the generated line in the rhyme list of the last word of the last input line
    """
    input_last = input_text.split(" ")[-1]
    generated_last = generated_text.split(" ")[-1]
    list_of_rhymes = pronouncing.rhymes(input_last)
    if print_list:
        print(list_of_rhymes)
    return (generated_last in list_of_rhymes) or (generated_last == input_last)


def check_rhyme_phyme(input_text, generated_text, include_consonant=False, print_list=False,
                      catch_spelling_error=True):
    """
    built based on CMU pronouncing; https://github.com/jameswenzel/Phyme
    better choice then check_rhyme_pronouncing
    :param catch_spelling_error:
    :param input_text: input text of the lyrics, could be one or two lines
    :param generated_text: output text of the lyrics, one line [subject to change]
    :return: a boolean indicate if the generated line in the rhyme list of the last word of the last input line
    :param include_consonant: consonant is the least rigorous type of rhyme; if false, will not include words here
    :param print_list: print the final list to compare with or not
    """
    input_text = re.sub(r"[^\w\d'\s]+", '', input_text)
    generated_text = re.sub(r"[^\w\d'\s]+",'', generated_text)

    input_last = input_text.split(" ")[-1]
    generated_last = generated_text.split(" ")[-1]

    # check if the last word is a number; if so, find its words equivalence
    if input_last.isnumeric():
        input_last = num2words(input_last)

    if generated_last.isnumeric():
        generated_last = num2words(generated_text)

    # catch spelling errors
    if catch_spelling_error:
        grammar_error = tool_en.check(input_last)
        if len(grammar_error) != 0:
            input_last = grammar_error[0].replacements[0]

    # get the groups
    if input_last.upper() in word_phone_dict:
        family = ph.get_family_rhymes(input_last)
        perfect = ph.get_perfect_rhymes(input_last)
        additive = ph.get_additive_rhymes(input_last)
        subtractive = ph.get_subtractive_rhymes(input_last)
        substitution = ph.get_substitution_rhymes(input_last)
        assonance = ph.get_assonance_rhymes(input_last)
        consonant = {}
        if include_consonant:
            consonant = ph.get_consonant_rhymes(input_last)

        union = set()
        for rhyme_dict in [family, perfect, additive, subtractive, substitution, assonance, consonant]:
            rhyme_lists = [x for sublist in rhyme_dict.values() for x in sublist]
            union = union | set(rhyme_lists)
            # print(len(union))

        if print_list:
            print(union)

        return (generated_last in union) | (generated_last == input_last)

    else:
        print("Rhyme word not in dictionary; skip the result.")
        return np.nan



# Similarity based evaluations.

def get_bleu_score(reference_text_list, generated_text, uniform_weight, ngram_order=None,
                   weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=None, auto_reweigh=False, use_token=True):
    """
    get bleu score;
    https://github.com/xzfxzfx/lyrics-generation/blob/main/evaluate.ipynb
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    :param reference_text_list: a list of reference sentences
    :param generated_text: generated line of lyrics
    :param uniform_weight: if set to true, the input weights will be overridden
    :param ngram_order: number of ngram order
    :param weights: length of the weights is the ngram order
    :param smoothing_function: method1-7; initiate using SmoothingFunction().method4 for example
    :param auto_reweigh: if auto reweigh or not
    :param use_token: if use token level or not
    :return: a bleu score
    """
    # split sentences into tokens
    references_tokens = []
    for sent in reference_text_list:
        references_tokens.append(sent.lower().split())

    generated_text_tokens = generated_text.lower().split()

    # set up weights based on ngram_order
    if uniform_weight:
        if ngram_order is None:
            ngram_order = int(len(generated_text_tokens) / 2)
        weights = [1 / ngram_order for i in range(ngram_order)]
        print(ngram_order, weights)

    # output results depending on if using tokens or not
    if use_token:
        print("References:\n", references_tokens)
        print("Candidate:\n", generated_text_tokens)
        score = sentence_bleu(references_tokens, generated_text_tokens,
                              weights=weights,
                              smoothing_function=smoothing_function,
                              auto_reweigh=auto_reweigh)
    else:
        print("References:\n", reference_text_list)
        print("Candidate:\n", generated_text)
        score = sentence_bleu(reference_text_list, generated_text,
                              weights=weights,
                              smoothing_function=smoothing_function,
                              auto_reweigh=auto_reweigh)
    return score


def get_summary_rouge_score(reference_text_list, generated_text, gram_type="LCS", score_type="f", summary_type="mean"):
    """
    get rouge score
    # https://github.com/xzfxzfx/lyrics-generation/blob/main/evaluate.ipynb
    :param summary_type: mean, max, min
    :param score_type: f for f score, p for precision, r for recall
    :param reference_text_list: a list of reference sentences
    :param generated_text: generated line of lyrics
    :param gram_type: unigram, bigram, or LCS(Longest Common Subsequence);
    :return: a rouge score
    """
    specific_scores = []
    specific_score = None

    for reference_text in reference_text_list:
        scores = rouge.get_scores(generated_text, reference_text)
        if gram_type == "unigram":
            specific_score = scores[0]["rouge-1"][score_type]
        if gram_type == "bigram":
            specific_score = scores[0]["rouge-2"][score_type]
        if gram_type == "LCS":
            specific_score = scores[0]["rouge-l"][score_type]

        specific_scores.append(specific_score)

    specific_scores = np.array(specific_scores)
    if summary_type == "mean":
        rouge_score = specific_scores.mean()

    if summary_type == "max":
        rouge_score = max(specific_scores)

    if summary_type == "min":
        rouge_score = min(specific_scores)

    return rouge_score


def get_summary_bert_score(reference_text_list, generated_text, summary_type="mean"):
    """
    bert based f1
    # https://github.com/xzfxzfx/lyrics-generation/blob/main/evaluate.ipynb
    # https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb
    :param summary_type: mean, max, min
    :param reference_text_list: a list of reference texts
    :param generated_text: generated line of lyrics
    :return: bert score (summary if there are multiple references)
    """
    f1s = []
    bert_score = None

    for reference_text in reference_text_list:
        p, r, f1 = score([reference_text], [generated_text], lang='en', verbose=True)
        f1s.append(f1.numpy()[0])

    f1s = np.array(f1s)
    if summary_type == "mean":
        bert_score = f1s.mean()

    if summary_type == "max":
        bert_score = max(f1s)

    if summary_type == "min":
        bert_score = min(f1s)

    return bert_score


def get_bert_plot(reference_text, generated_text):
    """
    get a bert plot
     # https://github.com/xzfxzfx/lyrics-generation/blob/main/evaluate.ipynb
    :param reference_text: a single line of reference text
    :param generated_text: generated line of lyrics
    :return: none; will show a similarity matrix plot
    """
    plot_example(generated_text, reference_text, lang="en")


# Diversity based evaluations.

def get_lexical_diversity(generated_text, compute_type="Simple TTR"):
    """
    get lexical diversity score
    # https://pypi.org/project/lexical-diversity/
    # https://arxiv.org/pdf/2006.14799.pdf
    :param generated_text: a line of generated lyrics, string
    :param compute_type:
        "HDD" (Hypergeometric distribution D),
        "Simple TTR (Type Token Ratio - # distinct tokens/# total tokens)",
        "Log TTR",
        "Mass TTR"
        "MSTTR" (Mean segmental TTR, default segment size is 50 words; customize using the window_length argument),
        "MATTR" (Moving average TTR default segment size is 50 words; customize using the window_length argument),
        "MABI" (Measure of lexical textual diversity - moving average, bi-directional)
    :return:
    """
    # tok = ld.tokenize(generated_text)
    flt = ld.flemmatize(generated_text)
    # print(flt)

    ld_score = None
    if compute_type not in ["HDD", "Simple TTR", "Log TTR", "Mass TTR", "MSTTR", "MATTR", "MABI"]:
        raise TypeError("Re-enter the type of lexical diversity")

    if compute_type == "HDD":
        ld_score = ld.hdd(flt)

    if compute_type == "Simple TTR":
        ld_score = ld.ttr(flt)

    if compute_type == "Log TTR":
        ld_score = ld.log_ttr(flt)

    if compute_type == "Mass TTR":
        ld_score = ld.maas_ttr(flt)

    if compute_type == "MSTTR":
        ld_score = ld.msttr(flt)

    if compute_type == "MATTR":
        ld_score = ld.mattr(flt)

    if compute_type == "MABI":
        ld_score = ld.mtld_ma_bid(flt)

    print(ld_score)
    return ld_score


# plagiarism.

def check_plagiarism(reference_text_list, generated_text):
    """
    check for complete sentence copying
    :param reference_text_list: a list of strings - references
    :param generated_text: a line of lyrics, string
    :return:
    """
    if generated_text in reference_text_list:
        return True
    else:
        return False


def run_evaluations(input_text, reference_text_list, generated_text):
    scores = {
        'grammar_score': get_grammar_score(generated_text, tool_en),
        'rhyme': check_rhyme_phyme(input_text, generated_text),
        'bleu_score': get_bleu_score(reference_text_list, generated_text, uniform_weight=True),
        'rouge_score': get_summary_rouge_score(reference_text_list, generated_text, summary_type="max"),
        'bert_score': get_summary_bert_score(reference_text_list, generated_text, summary_type="max"),
        'ld_score': get_lexical_diversity(generated_text),
        'plagiarism': check_plagiarism(reference_text_list, generated_text)
    }
    return scores


# def nn_perplexity(input_sent, target_sent, model):
#     input = char_tensor(input_sent)
#     target = char_tensor(target_sent)
#
#     hidden = model.init_hidden()
#     criterion = nn.CrossEntropyLoss()
#     loss = 0
#     for i in range(len(input_sent)):
#         output, hidden = model(input[i], hidden)
#     for i in range(len(target_sent)):
#         #  use hidden state from input sentence for generation
#         output, hidden = model(input[i], hidden)
#         loss += criterion(output, target[i].view(-1))
#     avg_loss = loss.item()/(len(target_sent))
#     return np.exp(avg_loss)
#
# def char_tensor(string):
#     tensor = torch.zeros(len(string)).long()
#     for c in range(len(string)):
#         tensor[c] = all_characters.index(string[c])
#     return Variable(tensor)