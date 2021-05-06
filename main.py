from model import *
from dataset import *
import torch.optim as optim
from evaluation import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

def gen_input(dataset, inp, target, artist, genre, word_lstm=False):
    ''' Takes in some lyric lines, converted to batchified tensors'''
    # print(inp)
    # print(target)
    word, char, ph = dataset.lyric_to_idx(inp)
    target = dataset.lyric_to_idx(target, False)

    if not word_lstm:
        artist, genre = dataset.get_artist_genre(len(word), artist, genre)
        features = [[artist], [genre]]
        input_list = [[word], [char], [ph], features, [target]]
    else:
        input_list = [[word], [target]]

    return input_list

def gen_multibatch_input(dataset, file_path ="data/csv/train.csv", subset = ["R&B"], num_lines = 2, batch_size = 3, word_lstm=False):
    ''' returns
        concatenated word representation: input list [words, chars, phones, features, target], inps, targets
        word lstm: input list [words, target], inps, targets'''
    word_list = []
    char_list = []
    ph_list = []
    target_list = []
    artist_list = []
    genre_list = []
    inps = []
    targets = []

    for _ in range(batch_size):
        inp, target, artist, genre, next_line = dataset.random_lyric_chunks(path="data/csv/train.csv", subset=subset,
                                                                            num_lines=num_lines)
        inps.append(inp)
        targets.append(target)
        word, char, ph = dataset.lyric_to_idx(inp)
        word_list.append(word)
        char_list.append(char)
        ph_list.append(ph)
        artist, genre = dataset.get_artist_genre(len(word), artist, genre)
        artist_list.append(artist)
        genre_list.append(genre)
        target = dataset.lyric_to_idx(target, False)
        target_list.append(target)

    if not word_lstm:
        features_list = [artist_list, genre_list]
        input_list = [word_list, char_list, ph_list, features_list, target_list]
        return input_list, inps, targets
    else:
        input_list = [word_list, target_list]
        return input_list, inps, targets


def decode_lyrics(dataset, tensor):
    ''' convert word tensor to string '''
    res = ''
    for idx in tensor:
        res += dataset.idx2word[idx.item()] + ' '
    return res

def batchify_sequence_labeling(input_batch_list, if_train=True):
    """
        input: list of words, chars and labels, various length. [words, chars, phones, features, target]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for one sentences, various length. (batch_size, sent_len, each_word_length)
            phones: phone ids one sentences, various length. (batch_size, sent_len, num_phone_in_word)
            features: genre and artist. (feature_nums, batch_size, sent_len)
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
    batch_size = len(input_batch_list[0])
    words = input_batch_list[0]
    chars = input_batch_list[1]
    phones = input_batch_list[2]
    features = input_batch_list[3]
    # print("features tensor: ",features)
    feature_num = len(features)
    # genres = [sent[2] for sent in input_batch_list]
    # artist = [sent[3] for sent in input_batch_list]
    target = input_batch_list[4]
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
        # print("target word: ", target_word)
        # print("seq len: ",seqlen)
        target_seq_tensor[idx, :seqlen] = torch.LongTensor(target_word)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idy][idx])
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
        for idy, (char, charlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            if charlen == 0:
                print("empty char list: ", chars)
            char_seq_tensor[idx, idy, :charlen] = torch.LongTensor(char)
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
    for idx, (seq, seqlen) in enumerate(zip(pad_phones, phone_seq_lengths)):
        for idy, (ph, phlen) in enumerate(zip(seq, seqlen)):
            if phlen == 0:
                print("empty phone list: ", phones)
            phone_seq_tensor[idx, idy, :phlen] = torch.LongTensor(ph)
    phone_seq_tensor = phone_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    phone_seq_lengths = phone_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    phone_seq_lengths, ph_perm_idx = phone_seq_lengths.sort(0, descending=True)
    phone_seq_tensor = phone_seq_tensor[ph_perm_idx]

    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    _, phone_seq_recover = ph_perm_idx.sort(0, descending=False)

    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
           char_seq_recover, phone_seq_tensor, phone_seq_lengths, phone_seq_recover, target_seq_tensor, mask

def batchify_word_list(input_list, if_train=True):
    """
        input: list of words, chars and labels, various length. [words, target]
            words: word ids for one sentence. (batch_size, sent_len)
            target: next word prediction. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            target_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_list[0])
    words = input_list[0]
    target = input_list[1]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    target_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()

    for idx, (seq, target_word, seqlen) in enumerate(zip(words, target, word_seq_lengths)):
        seqlen = seqlen.item()
        # fill first seq_len entries with each word_seq and target_seq
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        # print("target word: ", target_word)
        # print("seq len: ",seqlen)
        target_seq_tensor[idx, :seqlen] = torch.LongTensor(target_word)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    target_seq_tensor = target_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    return word_seq_tensor, word_seq_lengths, target_seq_tensor, mask

def calculate_loss(model, input_list, word_lstm=False):
    # if using concatenated word representation
    if not word_lstm:
        (word_inputs, feature_inputs, word_seq_lengths, word_seq_recover, char_inputs, char_seq_lengths,
         char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq, mask) = input_list
        artist_input = feature_inputs[0]
        genre_input = feature_inputs[1]
        ## model: WordSequence
        outs = model(word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover)
    else:
        (word_inputs, word_seq_lengths, target_seq, mask) = input_list
        ## model: WordLSTM
        outs = model(word_inputs, word_seq_lengths)

    batch_size = word_inputs.size(0)
    seq_len = word_inputs.size(1)

    loss_function = nn.NLLLoss(ignore_index=0)
    outs = outs.view(batch_size*seq_len, -1)
    # take the last word and compute loss
    # outs = outs[-1,].view(1,-1)
    score = F.log_softmax(outs, 1)
    total_loss = loss_function(score, target_seq.view(batch_size * seq_len))
    # total_loss = loss_function(score, target_seq.view(batch_size*seq_len)[-1].view(1))
    _, pred = torch.max(score, 1)
    pred = pred.view(batch_size, seq_len)
    # pred = pred.view(batch_size, 1)

    return total_loss, pred

def train(model, optimizer, dataset, spec_dict, num_lines = 2, batch_size=128, word_lstm=False):
    lr = 0.001
    # model = WordSequence(spec_dict, dataset)
    loss_list = []
    tot_loss = 0
    pred = None
    for i in range(spec_dict['iterations']):
        # set model in training mode
        model.train()
        model.zero_grad()
        # sample lines from dataset
        input_list, inps, targets = gen_multibatch_input(dataset,num_lines=num_lines, batch_size=batch_size, word_lstm=word_lstm)
        if not word_lstm:
            res = batchify_sequence_labeling(input_list, True)
        else:
            res = batchify_word_list(input_list, if_train=True)
        loss, pred = calculate_loss(model, res, word_lstm=word_lstm)
        tot_loss += loss.item()

        if (i+1) % spec_dict["plot_every"] == 0:
            loss_list.append(tot_loss/spec_dict["plot_every"])
            tot_loss = 0
        if (i+1) % spec_dict["print_every"] == 0:
            print("loss in current iteration: ", loss.item())
            print("Input string: ", inps[-1])
            print("Target string: ", targets[-1])
            print("prediced tensor: ", pred[-1])
            print("decoded string: ", decode_lyrics(dataset, pred[-1]))
        loss.backward()
        optimizer.step()
        model.zero_grad()

    return model, loss_list

def predict(model, dataset, num_lines = 2, gen_line = 1, data_path = "data/csv/train.csv", word_lstm=False):
    ''' takes in trained model, and generate gen_line numbers of new lines
        returns: predicted next line, target next line'''
    pred_next_line = ''
    print("Start predicting next line..")
    # TODO: gen_input should produce a new line as target if not in training, including new line character
    # TODO: remove artist in test data not in train data
    inp, target, artist, genre, next_line = dataset.random_lyric_chunks(path = data_path, subset=["R&B"], num_lines=num_lines, if_train=False)
    # input_list, inps, targets = gen_multibatch_input(dataset, num_lines=num_lines, batch_size=1,word_lstm=word_lstm)
    input_list = gen_input(dataset, inp, target, artist, genre, word_lstm=word_lstm)
    org_input = inp
    # print("prediction input: ", inp)
    # print("next line: ", target)
    if not word_lstm:
        res = batchify_sequence_labeling(input_list, if_train=False)
        (word_inputs, feature_inputs, word_seq_lengths, word_seq_recover, char_inputs, char_seq_lengths,
         char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq, mask) = res
    else:
        res = batchify_word_list(input_list, if_train=False)
        (word_inputs, word_seq_lengths, target_seq, mask) = res
    batch_size = word_inputs.size(0)
    seq_len = word_inputs.size(1)

    # set in eval mode
    model.eval()
    # feed into model, and generate until new line character, or allow 15 words max
    for _ in range(15):
        if not word_lstm:
            genre_input = feature_inputs[1]
            artist_input = feature_inputs[0]
            outs = model(word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths,
                         char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover)
        else:
            outs = model(word_inputs, word_seq_lengths)
        outs = outs.view(batch_size * seq_len, -1)
        score = F.log_softmax(outs, 1)
        _, pred = torch.max(score, 1)
        pred = pred.view(batch_size, seq_len)
        # print("prediction tensor: ", pred)
        pred = pred[0,-1].item()
        # find the corresponding word
        pred_word = dataset.idx2word[pred]
        # append predicted word
        pred_next_line  = pred_next_line + ' ' + pred_word
        # stop if next hit new line character
        # print(pred_word)
        # if pred_word == '\n':
        #     break
        # use new sentence as input to model
        inp = inp + ' '+pred_word
        target = inp # target word is not used in evaluation
        input_list = gen_input(dataset, inp, target, artist, genre, word_lstm=word_lstm)
        if not word_lstm:
            res = batchify_sequence_labeling(input_list, if_train=False)
            (word_inputs, feature_inputs, word_seq_lengths, word_seq_recover, char_inputs, char_seq_lengths,
             char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq, mask) = res
        else:
            res = batchify_word_list(input_list, if_train=False)
            (word_inputs, word_seq_lengths, target_seq, mask) = res
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

    return pred_next_line, next_line, org_input

def output_group_eval_scores(data_type, pred_dict):
    # calculate individual scores
    scores = {}
    for k in pred_dict:
        prediction, target, inp = pred_dict[k]
        scores[k] = run_evaluations(inp, [target], prediction)

    # aggregate scores
    evaluation_metrics = ["grammar_score", "rhyme", "bleu_score", "rouge_score", "bert_score", "ld_score",
                          "plagiarism"]
    sum_scores = {}
    for m in evaluation_metrics:
        sum_scores[m] = 0
        for k in scores:
            if scores[k][m]:
                sum_scores[m] += scores[k][m]

    for m in evaluation_metrics:
        sum_scores[m] = sum_scores[m] / len(scores)

    scores['mean'] = sum_scores

    # with open("outputs/evaluations_scores_" + data_type + ".txt", "w") as f_eval:
    #     json.dump(scores, f_eval, indent=4)

    return scores

if __name__ == "__main__":
    spec_dict = {"dropout": 0.5,
                 "num_lstm_layers": 1,
                 "bilstm_flag": True,
                 "word_bilstm_flag": False,
                 "use_artist": True,
                 "char_hidden_dim": 50,
                 "char_emb_dim": 50,
                 "char_model_type": "LSTM",
                 "word_emb_dim": 50,
                 "pre_train_word_embedding": None,
                 "feature_emb_dim": 50,
                 "final_hidden_dim": 512,
                 "learning rate": 0.001,
                 "iterations": 50,
                 "print_every": 20,
                 "plot_every": 25
                 }
    ds = Dataset('./data/csv/train.csv', subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    ds.load_dict()
    print("Successfully built dataset...")
    print("index of new line character: ", ds.word2idx['\n'])
    print("index of comma: ", ds.word2idx[','])
    print("index of <UNK>: ", ds.word2idx['<UNK>'])
    # input_list = gen_multibatch_input(ds, file_path='./data/csv/train.csv', subset=['R&B'], num_lines=2, batch_size=3)
    # print("batch size: ", len(input_list[0]))
    # tensors = batchify_sequence_labeling(input_list)
    # print(tensors[0])
    # ds.save_dict()
    # inp, target, artist, genre = ds.random_lyric_chunks(1)
    # input_list = gen_input(inp, target, artist, genre)
    # result = batchify_sequence_labeling(input_list)
    # print(result)

    model = WordSequence(spec_dict, ds)
    optimizer = optim.Adam(model.parameters(), lr=spec_dict['learning rate'])
    path = 'saved_data/wordchar2.1'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model, loss_list = train(model, optimizer, ds, spec_dict, num_lines=3, batch_size=50, word_lstm=False)
    path = 'saved_data/wordchar2.2'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)
    ## Predict using concatenated word representation model
    # word_lstm = False
    # prediction_examples = {}
    # for i in range(30):
    #     generated, target, input = predict(model, ds, data_path='data/csv/test_new.csv', word_lstm=word_lstm)
    #     prediction_examples[i] = [generated, target, input]
    # with open("outputs/predictions_examples_cat_word_lstm_test.txt", "w") as f_pred:
    #     json.dump(prediction_examples, f_pred, indent=4)
    with open("outputs/predictions_examples_cat_word_lstm_test.txt", "r") as f:
        prediction_examples = json.load(f)
    scores = output_group_eval_scores("cat_word_lstm_test", prediction_examples)
    print(scores)
    with open("outputs/evaluations_scores_cat_word_lstm_test.txt", "w") as f_eval:
        json.dump(scores, f_eval, indent=4)


    # model = WordLSTM(spec_dict, ds)
    # optimizer = optim.Adam(model.parameters(), lr=spec_dict['learning rate'])
    # path = 'saved_data/word_lstm2.4'
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model, loss_list = train(model, optimizer, ds, spec_dict, num_lines=3, batch_size=50, word_lstm=True)
    # path = 'saved_data/word_lstm2.4'
    # torch.save({'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()}, path)

    # plt.plot(loss_list)
    # plt.show()
    # word_lstm = True
    # prediction_examples = {}
    # for i in range(30):
    #     generated, target, input = predict(model, ds, data_path='data/csv/test_new.csv', word_lstm=word_lstm)
    #     prediction_examples[i] = [generated, target, input]
    # with open("outputs/predictions_examples_word_lstm_test.txt", "w") as f_pred:
    #     json.dump(prediction_examples, f_pred, indent=4)
    # with open("outputs/predictions_examples_word_lstm_test.txt", "r") as f:
    #     prediction_examples = json.load(f)
    # scores = output_group_eval_scores("word_lstm_test", prediction_examples)
    # print(scores)
    # with open("outputs/evaluations_scores_word_lstm_test.txt", "w") as f_eval:
    #     json.dump(scores, f_eval, indent=4)




