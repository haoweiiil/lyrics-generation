# Evaluating saved prediction results

## Instruction
Assuming prediction results are saved in txt files, we can use **"output_group_eval_scores"** function to compute the evaluation metrics for all generated lyrics. Our previously saved prediction files can be accessed in this [folder](https://drive.google.com/drive/folders/19s8hH6uYZ4V0Dgsv36rIxwobbBcSEeZD?usp=sharing). Run the following script in python console, and evaluation results will be saved in "outputs/evaluations_scores_**model_name**.txt", where model_name can be "baseline_model", "exp_lstm_model", or "gpt2_model". This is specified via variable "eval_path".

The following script can be used for all model prediction result by changing "model_pred_path" to reflect desired model result. 
The prediction files can be found in this Google Drive folder: [https://drive.google.com/file/d/18iJ_BLGWrElIfJtoZbVCK_nHRIWBE0E2/view?usp=sharing](https://drive.google.com/file/d/18iJ_BLGWrElIfJtoZbVCK_nHRIWBE0E2/view?usp=sharing). Replace model_pred_path with the appropriate link if applicable.

    ## Evaluating baseline model
    model_pred_path = "outputs/predictions_baseline.txt"
    eval_path = "baseline_model"
    
    ## Evaluating expanded LSTM model
    model_pred_path = "outputs/predictions_exp_lstm.txt"
    eval_path = "exp_lstm_model"
    
    ## Evaluating fine-tuned GPT model
    model_pred_path = "outputs/predictions_examples_gpt2_model.txt"
    eval_path = "gpt_model"
    
    with open(model_pred_path, "r") as f:
        prediction_examples = json.load(f)
    scores = output_group_eval_scores(eval_path, prediction_examples)
    # print average evaluation scores
    print(scores["mean"])
    
    
    
    
    
    
## Example
* Input: 
