# Instructions on running models

For LSTM models discussed below, we have saved the model parameters, and therefore it's not necessary to rerun the training process again. However, we do provide code to re-train the models from scratch below.

## LSTM Models

### I. Loading saved model

1. First, please download the [saved model](https://drive.google.com/drive/folders/1VlpxPyEnoXo9VgdSMvubMa9WP0bvd2fu?usp=sharing) to folder *saved_model*. Then run the following script in Python console.

2. Then make sure the dataset dictionaries are saved in folder *saved_data*; they can downloaded from [Google Drive](https://drive.google.com/drive/folders/1vbUooe7-E7rltR5wMpJNN7lTDUn62afQ?usp=sharing).


        from main import *
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
                     "iterations": 100,
                     "print_every": 20,
                     "plot_every": 25
                     }
        ds = Dataset('./data/csv/train.csv', subset=['R&B'])
        ds.load_dict()
    
        # Loading baseline LSTM model             
        baseline_model = WordLSTM(spec_dict, ds)
        path = 'saved_model/word_lstm2.4'
        checkpoint = torch.load(path)
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Loading Expanded LSTM model
        exp_lstm_model = WordSequence(spec_dict, ds)
        path = 'saved_model/wordchar2.2'
        checkpoint = torch.load(path)
        exp_lstm_model.load_state_dict(checkpoint['model_state_dict'])
 
    
### II. Making next-line predictions

The following script samples from test file, and produce next-line generation given sampled lyrics input. The number of generation can be changed via variable *num_iters*. The generation results are recorded in a dictionary with iteration number as key, and **\[Input lyrics, Target lyric, Generated lyric\]** as value. The results are configured to be saved in a text file under *outputs* folder.

To switch between the baseline and expanded version of LSTM model, simply change the configurations of variable **word_lstm**, **model**, and **eval_out_path** using the following script. 



    ## Predicting using baseline LSTM model
    word_lstm = True
    model = baseline_model
    eval_out_path = "outputs/predictions_baseline.txt"
    
    ## Predicting using expanded LSTM model
    # word_lstm = False
    # model = exp_lstm_model
    # eval_out_path = "outputs/predictions_exp_lstm.txt"
    
    prediction_examples = {}
    num_iters = 30
    for i in range(num_iters):
        generated, target, input = predict(model, ds, data_path='data/csv/test_new.csv', word_lstm=word_lstm)
        prediction_examples[i] = [generated, target, input]
    with open(eval_out_path, "w") as f_pred:
        json.dump(prediction_examples, f_pred, indent=4)

Alternatively, the *main.py* script is currently configured to load baseline model, generate prediction and print evaluation result. The following command line should initiate the process. We would advise against re-starting the training process since it would take quite some time, and instead run the saved models. 
    
    $ python main.py 
     
    
    
## Fine-tuned GPT-2
Project folder link: [https://drive.google.com/drive/folders/1ouFWNXlFdGqnj2I8SBaLCVmWajoYoTNP?usp=sharing](https://drive.google.com/drive/folders/1ouFWNXlFdGqnj2I8SBaLCVmWajoYoTNP?usp=sharing)

Fine-tuned GPT-2 Colab link: [https://colab.research.google.com/drive/1gHb16zM-fJURkVB2dB4_RppBS7cOLnxo?usp=sharing](https://colab.research.google.com/drive/1gHb16zM-fJURkVB2dB4_RppBS7cOLnxo?usp=sharing)

### I. Set up
Run all cells in the colab file before Section 1.

### II. Loading saved model
Section 1-3 includes training the model and examining the loss of the model.
Section 4 load saved model to the notebook and generate text. Run the whole section to generation new lyrics. Last two cells under secion 4.1 is commented out to avoid overiding our current results.
