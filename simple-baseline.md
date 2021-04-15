# Simple Baseline

## Baseline Model Description
We implement n-gram model as our baseline. We referred to n-gram model implementation from [HW3](http://markyatskar.com/cis530_sp2021/homework/ngram-lms/ngram-lms.html). In training our n-gram model, we use similar data sampling technique as training neural models: we randomly generate a starting line index and select the following lines as one sample, and then we update the n-gram model accordingly. We use random sampling instead of inputting the entire lyric corpus at once, because in this way, the model can learn different ways to start generating new lyrics. And given the size of our input data, we believe given sufficient iteration, this baseline model would be able to learn the patterns in a variety of genres produced by different artists. 

We built model on n = \[6, 7, 8\]; we found in our previous homework that n-gram produces reasonable generation outcome given these specifications. And we draw 2000 samples of 20 lines as training input.

 ## Generation Examples
 
 
 ## Evaluation Scores 
 
 ### Perplexity