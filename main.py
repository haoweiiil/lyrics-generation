from model import *
from dataset import *
import torch.optim as optim
import matplotlib.pyplot as plt

def predict(model, dataset, num_lines = 2, gen_line = 1, data_path = "data/csv/train.csv"):
    ''' takes in trained model, and generate gen_line numbers of new lines
        returns: predicted next line, target next line'''
    pred_next_line = ''
    print("Start predicting next line..")
    # TODO: gen_input should produce a new line as target if not in training, including new line character
    # TODO: remove artist in test data not in train data
    inp, target, artist, genre, next_line = dataset.random_lyric_chunks(path = data_path, subset=["R&B"], num_lines=num_lines, if_train=False)
    print("prediction input: ", inp)
    input_list = gen_input(dataset, inp, target, artist, genre, if_train=False)
    res = batchify_sequence_labeling(input_list, False)
    (word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths,
     char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq_tensor, mask) = res
    batch_size = word_seq_tensor.size(0)
    seq_len = word_seq_tensor.size(1)

    # set in eval mode
    model.eval()
    # feed into model, and generate until new line character, or allow 15 words max
    for _ in range(15):
        genre_input = feature_seq_tensors[1]
        artist_input = feature_seq_tensors[0]
        outs = model(word_seq_tensor, genre_input, artist_input, word_seq_lengths, char_seq_tensor, char_seq_lengths, char_seq_recover,
                ph_inputs, ph_seq_lengths, ph_seq_recover)
        outs = outs.view(batch_size * seq_len, -1)
        score = F.log_softmax(outs, 1)
        _, pred = torch.max(score, 1)
        pred = pred.view(batch_size, seq_len)
        print("prediction tensor: ", pred)
        pred = pred[0,-1].item()
        # find the corresponding word
        pred_word = dataset.idx2word[pred]
        # append predicted word
        pred_next_line  = pred_next_line + ' ' + pred_word
        # stop if next hit new line character
        print(pred_word)
        if pred_word == '\n':
            break
        # use new sentence as input to model
        inp = inp + ' '+pred_word
        target = inp # target word is not used in evaluation
        print("full word")
        input_list = gen_input(dataset, inp, target, artist, genre, if_train=False)
        res = batchify_sequence_labeling(input_list, False)
        (word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths,
         char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq_tensor, mask) = res
        batch_size = word_seq_tensor.size(0)
        seq_len = word_seq_tensor.size(1)
        print("input len to lstm generation: ", seq_len)

    return pred_next_line, next_line

def gen_input(dataset, inp, target, artist, genre, batch_size = 1, if_train=True):
    ''' Takes in dataset object, return randomly sampled lines, converted to batchified tensors'''
    batches = []
    # TODO: handle batch_size > 1
    # print(inp)
    # print(target)
    word, char, ph = dataset.lyric_to_idx(inp)
    input_list = [[word], [char], [ph]]
    features = [dataset.get_artist_genre(len(word), artist, genre)]

    target = [dataset.lyric_to_idx(target, False)]
    input_list.append(features)
    input_list.append(target)

    return input_list

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
    batch_size = len(input_batch_list[0])
    words = input_batch_list[0]
    chars = input_batch_list[1]
    phones = input_batch_list[2]
    features = np.array(input_batch_list[3])
    # print("features tensor: ",features)
    feature_num = len(features[0][0])
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
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
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

def calculate_loss(model, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                   char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover, target_seq, mask):
    artist_input = feature_inputs[0]
    genre_input = feature_inputs[1]
    batch_size = word_inputs.size(0)
    seq_len = word_inputs.size(1)

    outs = model(word_inputs, genre_input, artist_input, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, ph_inputs, ph_seq_lengths, ph_seq_recover)
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

def train(model, dataset, spec_dict, num_lines = 2):
    lr = 0.001
    # model = WordSequence(spec_dict, dataset)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    tot_loss = 0
    pred = None
    for i in range(spec_dict['iterations']):
        # set model in training mode
        model.train()
        model.zero_grad()
        # sample lines from dataset
        # inp, target, artist, genre, next_line = dataset.random_lyric_chunks(path = "data/csv/train.csv", subset=["R&B"], num_lines=num_lines)
        inp, target, artist, genre = dataset.random_song(path = "data/csv/train.csv", subset=["R&B"], num_lines=5)
        input_list = gen_input(dataset, inp, target, artist, genre)
        res = batchify_sequence_labeling(input_list, True)
        (word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths,
         char_seq_recover, phone_seq_tensor, phone_seq_lengths, phone_seq_recover, target_seq_tensor, mask) = res
        loss, pred = calculate_loss(model, res[0],res[1],res[2],res[4],res[5],res[6],res[7],res[8],res[9],res[10], res[11])
        tot_loss += loss.item()

        if (i+1) % spec_dict["plot_every"] == 0:
            loss_list.append(tot_loss/spec_dict["plot_every"])
            tot_loss = 0
        if (i+1) % spec_dict["print_every"] == 0:
            print("loss in current iteration: ", loss.item())
            print("Input string: ", inp)
            print("Target string: ", target)
            print("prediced tensor: ", pred)
            print("decoded string: ", decode_lyrics(dataset, pred[0]))
        loss.backward()
        optimizer.step()
        model.zero_grad()

    return model, loss_list

if __name__ == "__main__":
    spec_dict = {"dropout": 0.7,
                 "num_lstm_layers": 2,
                 "bilstm_flag": True,
                 "word_bilstm_flag": False,
                 "use_artist": True,
                 "char_hidden_dim": 128,
                 "char_emb_dim": 100,
                 "char_model_type": "LSTM",
                 "word_emb_dim": 256,
                 "pre_train_word_embedding": None,
                 "feature_emb_dim": 128,
                 "final_hidden_dim": 512,
                 "learning rate": 0.001,
                 "iterations": 2500,
                 "print_every": 250,
                 "plot_every": 50
                 }
    ds = Dataset('./data/csv/train.csv', subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    print("Successfully built dataset...")
    print("index of new line character: ", ds.word2idx['\n'])
    print("index of comma: ", ds.word2idx[','])
    print("index of <UNK>: ", ds.word2idx['<UNK>'])
    # inp, target, artist, genre = ds.random_lyric_chunks(1)
    # input_list = gen_input(inp, target, artist, genre)
    # result = batchify_sequence_labeling(input_list)
    # print(result)

    model = WordSequence(spec_dict, ds)
    model, loss_list = train(model, ds, spec_dict, num_lines=2)
    torch.save(model.state_dict(), 'model3.pt')
    plt.plot(loss_list)
    plt.show()
    pred_lines = predict(model, ds)
    print(pred_lines)



