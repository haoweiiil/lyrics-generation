import random
import string
import numpy as np
import eng_to_ipa
import cmudict
import language_tool_python
from nltk.tokenize.regexp import RegexpTokenizer
from num2words import num2words

# train_lyrics_list = pd.read_csv("../data/song_lyrics/train_data.csv")['Lyrics'].values  # list of song lyrics
tool_en = language_tool_python.LanguageTool("en-US")


def clean_lyrics(text, line_sep=" \n "):
    text = text.replace("\n\n", line_sep).replace("\n", line_sep)
    text_list = text.split(line_sep)
    text_list = [x.lstrip() for x in text_list]
    text = line_sep.join(text_list)
    return text


def format_lyrics(lyrics_list, output_type, end_sep="<end>", line_sep=" \n "):
    lyrics_list = [x.lstrip() for x in lyrics_list]

    if output_type == "string":
        lyrics_string = end_sep.join(lyrics_list)  # turn the lyrics to a string with separator "<end>"
        clean_lyrics_string = clean_lyrics(lyrics_string, line_sep)
        return clean_lyrics_string

    if output_type == "list of lyrics":
        lyrics_string = end_sep.join(lyrics_list)  # turn the lyrics to a string with separator "<end>"
        clean_lyrics_string = clean_lyrics(lyrics_string, line_sep=line_sep)
        clean_lyrics_list = clean_lyrics_string.split(end_sep)
        return clean_lyrics_list

    if output_type == "list of lines":
        lyrics_string = (end_sep + "\n").join(lyrics_list) + end_sep
        clean_lyrics_string = clean_lyrics(lyrics_string)
        clean_line_list = clean_lyrics_string.split(" \n ")
        return clean_line_list
    else:
        raise ValueError


def sample_lines(lyrics, n):
    """
    sample lines from a lyrics
    :param lyrics: cleaned lyrics with separator " \n "
    :param n: number of lines to sample from
    :return:
    """
    lyrics_lines = lyrics.split(" \n ")
    i = random.randint(0, max(len(lyrics_lines) - n, 0))
    sample = lyrics_lines[i:i + n]
    return " \n ".join(sample) + " \n "


def sample_lyrics_with_author_song_names(lyrics_df, n_lyrics, sample_n_lines=None, break_into_lines=False):
    """
    sample lyrics from a list of lyrics
    :param lyrics_df: df of lyrics with author and song information
    :param n_lyrics: number of lyrics to sample
    :param sample_n_lines: number of lines from the lyrics to sample
    :param break_into_lines: if we want to break each song from string to list using separator " \n "
    :return: selected lyrics as a list of string, or list of lists of lines
    """
    all_lyrics = lyrics_df['Lyrics'].values
    all_authors = lyrics_df['Artist'].values
    all_songs = lyrics_df['Song'].values
    all_genres = lyrics_df['Genre'].values
    n_lyrics = min(len(all_lyrics), n_lyrics)

    # randomly select a lyrics index
    idx = np.arange(len(all_lyrics))
    selected_idx = np.array([np.random.choice(idx) for i in range(n_lyrics)])
    selected_lyrics_raw = all_lyrics[selected_idx]
    selected_authors = all_authors[selected_idx]
    selected_songs = all_songs[selected_idx]
    select_genres = all_genres[selected_idx]

    # clean each song in the lyrics list
    selected_lyrics = [clean_lyrics(x) for x in selected_lyrics_raw]

    # sample if applicable
    if sample_n_lines is not None:
        selected_lyrics = [sample_lines(x, sample_n_lines) for x in selected_lyrics]

    # split each song into list of lines if applicable
    if break_into_lines:
        selected_lyrics = [x.split(" \n ") for x in selected_lyrics]
    return selected_lyrics, selected_authors, selected_songs, select_genres


# Tokenizers.

def nltk_tokenizer_no_punctuation(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sent)


def custom_tokenizer(sent):
    tokens = [word.strip(string.punctuation) for word in sent.split(" ")]

    while "" in tokens:
        tokens.remove("")

    # while "\n" in tokens:
    #     tokens.remove("\n")

    return tokens


# Embeddings.

def get_tfidf(lyrics_list, tokenizer, ignore_capital=True, return_dict=True):
    """
    tfidf embedding on the word level
    :param lyrics_list:
    :param tokenizer: which tokenizer to use
    :param return_dict: return dictionary or not
    :param ignore_capital: True or False
    :return: tfidf of the words using all lyrics
    """
    if ignore_capital:
        lyrics_list = [x.lower() for x in lyrics_list]

    all_lyrics_string = clean_lyrics("".join(lyrics_list))

    tokens = tokenizer(all_lyrics_string)
    unique_words = set(tokens)
    word2idx = {w: i + 1 for i, w in enumerate(unique_words)}

    # term frequency
    tf = np.zeros((len(unique_words), len(lyrics_list)))
    for i in range(len(lyrics_list)):
        song_string = clean_lyrics(lyrics_list[i])
        song_tokens = tokenizer(song_string)
        for word in song_tokens:
            tf[word2idx[word] - 1][i] += 1

    # idf and tfidf
    idf = np.log(len(lyrics_list) + 0.1, np.count_nonzero(tf, 1) + 0.1).reshape(-1, 1)
    tfidf = tf * idf

    tfidf_output = tfidf

    if return_dict:
        tfidf_output = {}
        for word in word2idx:
            tfidf_output[word] = tfidf[word2idx[word] - 1]

    return tfidf_output


# Others.
def phone(string_word):  # phonemes
    return eng_to_ipa.ipa_list(string_word)[0]


def get_phones(word):
    return cmudict.dict()[word]


def add_padding(uneven_list, pad_to=None):
    lengths = [len(x) for x in uneven_list]
    for i in range(len(uneven_list)):
        if pad_to is None:
            pad_to = max(lengths)
        uneven_list[i] = np.pad(uneven_list[i], (0, pad_to - lengths[i]), "constant", constant_values=0)
    return np.array(uneven_list)


def get_grammar_top_replacement(text, tool, output_error_matches=False):
    grammar_error = tool.check(text)
    if len(grammar_error) != 0:
        if len(grammar_error[0].replacements) != 0:
            print(text)

            if output_error_matches:
                for match in grammar_error:
                    print(match.ruleId, match.replacements)

            text = grammar_error[0].replacements[0]
            print("replace input last with", text)
    return text


def clean_word(word):
    # number
    if word.isnumeric():
        word = num2words(word)
    # grammar/spelling errors
    word = get_grammar_top_replacement(word, tool_en)
    return word


if __name__ == '__main__':
    pass
    # train_data = pd.read_csv("../data/song_lyrics/train_data.csv")
    # print(sample_lyrics_with_author_song_names(train_data, 1, 10))
    # print(get_grammar_top_replacement("fucked", tool_en, output_error_matches=True))  # corner case

