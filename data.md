# Data Description

## Data Source
We downloaded our dataset from a Kaggle dataset named [Multi-Lingual Lyrics for Genre Classification](https://www.kaggle.com/mateibejan/multilingual-lyrics-for-genre-classification). Since the purpose of our analysis is not classification, we are not using the train/test split provided directly. 

Instead we use only the training dataset and performed further data cleaning described below.

Here is a snippet of the data: ![data example 1](/images/preprocessed_df.png) 

## Data Processing 

1. We select only English lyrics identified in the "Language" column. We further restrict language to English by running python language detection package, [*cld3*](https://pypi.org/project/pycld3/), on each row. We only keep those lyrics identified as "en" and marked as "is_reliable" by *cld3*.  

2. We also noticed that there are some song lyrics that don't have line breaks. We suspect this is due to error in data collection, and thus we remove entries without "\n" character in lyrics.

3. We dropped duplicates on subset of "Artist", "Song", and "Genre" to eliminate repeated information in lyrics that might only differ in lyrics formatting. 

4. We removed lyrics with excess usage of non-alphabetic characters. Specifically we removed lyrics containing "---------".

5. Finally, since we are only interested in the content of lyrics, and not structural information, such as whether this paragraph is chorus or not, we replaced phrases between brackets and including brackets with space. So phrases like "\[chorus 1\]" will be erased from lyric.

## Final Data Format

After processing, we have total of 192,123 songs. We divide this dataframe into train, dev and test data. The corresponding data are saved into csv.

We keep information regarding genre and artist in the csv files since they might be needed in future model implementation.