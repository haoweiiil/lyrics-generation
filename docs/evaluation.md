# Evaluating saved prediction results

Assuming prediction results are saved in txt files, we can use **"outputs/evaluations_scores_cat_word_lstm_test.txt"** function in *evaluations.py* script to compute the evaluation metrics for all generated lyrics. Run the following script in python console, and evaluation results will be saved in specified "result_path".

The following script can be used for all model prediction result by changing "model_pred_path" to reflect desired model result. 

    with open(model_pred_path, "r") as f:
        prediction_examples = json.load(f)
    scores = output_group_eval_scores("baseline_model", prediction_examples)
    result_path = "outputs/evaluations_scores_cat_word_lstm_test.txt"
    with open(result_path, "w") as f_eval:
        json.dump(scores, f_eval, indent=4)