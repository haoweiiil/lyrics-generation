# Simple Baseline

## Baseline Model Description
We implement n-gram model as our baseline. We referred to n-gram model implementation from [HW3](http://markyatskar.com/cis530_sp2021/homework/ngram-lms/ngram-lms.html). In building our baseline model, we chose two type of genre \(*R&B* and *Indie*\) as input to understand if our model would exhibit different patterns if trained on separate genres. We then evaluated the model given test data of corresponding genre.   

<!---
use similar data sampling technique as training neural models: we randomly generate a starting line index and select the following lines as one sample, and then we update the n-gram model accordingly. We use random sampling instead of inputting the entire lyric corpus at once, because in this way, the model can learn different ways to start generating new lyrics. And given the size of our input data, we believe given sufficient iteration, this baseline model would be able to learn the patterns in a variety of genres produced by different artists. 
-->

We built model on n = \[5, 6, 7\]; we found in our previous homework that n-gram produces reasonable generation outcome given these specifications. We also add smoothing parameter *k = 0.0000001*. We found that with a smoothing parameter too large, the model would generate insensible sequence of characters, which is probably because the variety of characters following certain context is relatively small compared to model's vocab size, and if k were too big, then it would assign larger probability to some unseen characters following some context.  

 ## Generation Examples
 |       | 5-Gram | 6-Gram | 7-Gram |
|-------|--------|--------|--------|
| **R&B**   | I I could begin though the more, I don't be respecial will be back to you was fade<br />wear these people the jamming about<br />Whatever for the bit on me<br /><br />Why'd givine and i for you gotta right<br /><br />If your mind on, blame<br />Than stand war<br />On 02 14 03.|The heal me<br />tell you better so long to be<br />I've giveness, hey<br />i hope<br />Livin? insided the actressing<br />let me<br />Please your fake their wish this is far<br />the found and sixty-two<br />And late, yeah, that you<br />I don't even competitionally calmed this is hold out, no no)|She was<br />If I was broken home<br />if she'd everybody's take a watchin' for, 2004.<br />And you no matter<br />summertime, one that day or no biblical like when we used to be love a window you by my heartache, what your name our Savior comes to aching hurts me<br />He healer without speaks unto you as hard and the wearing a wish I want to give<br />Oh, I love with you<br />why don't know God made something your proper.<br />It's so caliente<br />you didn't fuck bitch black and see you   |
| **Indie** | You're down to<br />Take you'll love, got me over knew<br />That the eyes with my girls are wing a his fat glove your arc<br />Porce sacrawling. all my hello, we'll looking<br />Gossibles and Your voice the mares<br /><br />They clean maybe than I can't way I don't sleep beliefs<br />This horse on come countain<br /><br />But our life could god sprince of ourself| Sweet apology<br />Then bad line of these world whiskey and sleep 'cause you're nothings that's wrong (shout<br />You thing's the wet myself out for You<br /><br />My little more electric with dreamed Sadie<br />Yeah, you weren't wear I've that bring is up<br />Rider of us don't be a funk that family.<br />The airport to hide you<br />(Verse)<br />I wrote and stuffed in the way I hear |Greasy here <br />What a Swedish generation?<br /><br />Am I repeated)<br />I luv ya (layered little now<br />Oh, you must be having now<br />I'll take my past if for the ooooooooooo<br />there's a lesson to the need to take<br />we're both breadcrumbs<br />And you could ask what's all you|
 
 By manually inspecting the generated text, we notice the following:
 * Overall, n-gram is capable of learning of the general lyrics structure, i.e., frequent line breaks and short paragraphs. 
 * In most cases, it has learnt that each line should start with capital letters. But again, format in lyrics is more fluid than normal languages, so the pattern might not have been consistent in the training data either. 
 * Generally, n-gram model is able to learn individual vocabulary. We see typos here and there, but it is doing a decent job at generating words. However, the connection between words seem weak: the word sequences do not make sense beyond the local context. In other words, vocabularies only care about being consistent with the immediately adjacent words. And this is exactly the weakness of n-gram, especially on a character-level.
 * Vocabulary used in Indie songs seem to be softer, and they all seem to revolve around *love*, which is probably a shared trait across all genre. 

 
 ## Evaluation Scores 
 In our main analysis, we are mostly concerned with next line lyrics prediction as a measurement of how well our model is understanding the main theme of songs and if it were able to correctly predict the content coming up next.  The ability of n-gram model to predict next line of lyrics based on previous content is understandably limited by length of context in use. But we demonstrate here the performance of simple baseline model, and later compare these scores to more complex neural models.

We use R&B data 7-gram model for next line generation. In the table below are several examples with input data from training data and test data, and their corresponding evaluation metrics score.

1. Training Example 1
    * Input Lyric (Previous lyric): *The more I know, the less I understand*
    * Target Lyric (Real next line): *And all the things I thought I'd figured out, I have to learn again* 
    * Genereate Lyric: *It burns*
    
2. Training Example 2
    * Input Lyric (Previous lyric): *From the rain and storm*
    * Target Lyric (Real next line): *And even when my enemies pursue me* 
    * Genereate Lyric: *But you, the cost us back and girl this mark, no love*

3. Test Example 1
    * Input Lyric (Previous lyric): *What's your name?*
    * Target Lyric (Real next line): *Girl what's your number?* 
    * Genereate Lyric: *Jesus, just to me*

4. Test Example 2
    * Input Lyric (Previous lyric): *This is Christmas, let the world sing*
    * Target Lyric (Real next line): *Let us all begin to heal* 
    * Genereate Lyric: *Stays burning back to your tabernacle*

We present below the evaluation score computed using the example above. Additionally we evaluated the model on randomly selected test lyrics tuples, and computed the average score, also shown in the table below. For boolean metrics, the average computes the proportion of result that are true.

|               | Train 1 | Train 2 | Test 1 | Test 2 | Avg. Test |
|---------------|---------|---------|--------|--------|-----------|
| Grammar score | 0       | 0       | 0      | 0      | 0.028     |
| Rhyme         | False   | False   | False  | False  | 0         |
| Bleu score    | 0       | 2.773   | 0      | 4.354  | 3.335     |
| Rouge Score   | 0       | 0       | 0      | 0.167  | 0.044     |
| Bert Score    | 0.825   | 0.830   | 0.827  | 0.821  | 0.814     |
| Ld score      | 1.0     | 1.0     | 1.0    | 1.0    | 0.987     |
| Plagiarism    | False   | False   | False  | False  | 0         |
