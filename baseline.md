# Baseline model: word-level LSTM

As we've discussed in previous submission that, although language generation task has been widely studied, there hasn't been a shared task created for lyric generation. Therefore, we've drawn insights from previous studies on lyrics prediction, as well as other forms of generation. For our published baseline implementation, we choose the word-based LSTM model published by [Tikhonov and Yamshchikov (2018)](https://arxiv.org/pdf/1807.07147.pdf). We'll provide an introduction to this paper below.

## Related Paper and Model Implementation

*Please note that we mentioned this paper in our previous literature review. Thus part of the following paragraphs borrows from our project guideline submission.*

**Tikhonov and Yamshchikov (2018)** in their "Guess who? Multilingual Approach For The Automated Generation Of Author-Stylized Poetry" discussed an author-stylized poetry generation mechanism based on LSTM neural network, which produces poem of higher relevance to the target author The authors argue that model trained on large corpus of texts that comprises of different styles might not learn to produce congruent and single-style sentences normally found in a single passage. To address this style related issue, this study trains model on entire corpus but only evaluates model performance conditional on target artists. We also found that model architecture employed by this study to be quite relevant and applicable to our analysis, as shown in figure below. 
![baseline model](/images/baseline%20model%20architecture.png) 
It embeds each word with char-level and phoneme information, as well as supplements document-level embedding at every step to prevent the model from forgetting general document pattern in longer sequences. As for evaluation of generated text, Tikhonov and Yamshchikov first sampled a random starting line from target author's works, then calculated BLEU score between the three next actual lines and generated texts. Results show that their full model provides a 1.7\% boost to a vanilla LSTM model. 

We chose the model published in this paper as baseline as we believe it presents the right amount of perplexity for a strong baseline model. On top of vanilla word-level LSTM model, this design also incorporates more granular level bidirectional LSTM with character-level and phoneme-based information, which we believe could provide richer context to generation, and could better handle unseen words in testing input. Additionally, on document level, this paper uses author and document embedding. We made some modifications that we believe are appropriate for our application: firstly we keep the char-level and phoneme-level implementation, but instead of document embedding, we use genre embedding, since in prediction, we wouldn't have access to information of the full passage. We then train the model on randomly selected lyrics chunks. We predict the next word using some sampled input, and continue generating future words by appending previously generated word to input lyric. We will discuss the result below.

## How to run

There are three files relevant to this baseline implementation: *dataset.py*, *model.py*, and *main.py*. We create an object Dataset in dataset.py that records information about the training input, and supports random lyric chunks sampling, and coverts strings to index representation. We construct our final word-level LSTM model using class **WordSequence** in model.py. It takes concatenated word representations and run through LSTM layer with a sequence of input. We borrow heavily from **NCRF++** package's implementation, available publicly on [Github](https://github.com/jiesutd/NCRFpp). Finally in main.py, we convert input to appropriate tensor format, and train the model. 

Hyperparameters and training process are recorded in main function in *main.py*. To load saved model, and generate prediction, run the following code:

    from nltk import word_tokenize
    spec_dict = {"dropout": 0.5,
                 "num_lstm_layers": 1,
                 "bilstm_flag": True,
                 "final_bilstm_flag": True,
                 "use_artist": True,
                 "char_hidden_dim": 128,
                 "char_emb_dim": 100,
                 "char_model_type": "LSTM",
                 "word_emb_dim": 128,
                 "pre_train_word_embedding": None,
                 "feature_emb_dim": 128,
                 "final_hidden_dim": 512,
                 "iterations": 8000,
                 "print_every": 50,
                 "plot_every": 50}
    # path = './data/csv/train.csv'
    ds = Dataset(path, subset=['R&B'])
    vocab_size = ds.tokenize_corpus(word_tokenize)
    print("Successfully built dataset...")
    ## Make sure all the pickled dictionaries are in a folder called "saved_data" in the main directory
    ds.load_dict() # load from saved dictionaries
    model = WordSequence(spec_dict, ds)
    model_path = "model.pt"
    model.load_state_dict(torch.load(model_path))
    pred_lines, target_line = predict(model, ds, data_path = "data/csv/train.csv")
    print(pred_lines)
    print(target_line)
    
A few things to note:
1. "path" variable points to a csv input file, can be accessed from [Google Drive](https://drive.google.com/drive/folders/1i0LbMcoSYLQ4QjXRr5-zsnKDe7Skd32W?usp=sharing).
2. Please make sure all the pickled dictionaries are in a folder called "saved_data" in the main directory. These dictionaries can be downloaded [here](https://drive.google.com/drive/folders/1vbUooe7-E7rltR5wMpJNN7lTDUn62afQ?usp=sharing).
3. "model_path" refers to the file path of saved model.
4. This chunk of code will produce the next line given randomly sampled lyrics, where "target_line" is the true next line, and "pred_line" is generated from our model

## Evaluation result

We evaluated the baseline model with similar procedure as the previous simple-baseline version. We randomly sample input lines from training and test files, then predict the next line, and compare this line to true next line using various metrics.

The result is presented in the following table. 

**TODO: Change the numbers in the following table**

|               | Train 1 | Train 2 | Test 1 | Test 2 | Avg. Test |
|---------------|---------|---------|--------|--------|-----------|
| Grammar score | 0       | 0       | 0      | 0      | 0.028     |
| Rhyme         | False   | False   | False  | False  | 0         |
| Bleu score    | 0       | 2.773   | 0      | 4.354  | 3.335     |
| Rouge Score   | 0       | 0       | 0      | 0.167  | 0.044     |
| Bert Score    | 0.825   | 0.830   | 0.827  | 0.821  | 0.814     |
| Ld score      | 1.0     | 1.0     | 1.0    | 1.0    | 0.987     |
| Plagiarism    | False   | False   | False  | False  | 0         |
