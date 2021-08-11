# Lyrics Generation using N-gram, LSTM and GPT-2 Models

*Note: this project was created for Computation Linguistics course at UPenn*

## Project Introduction

Lyrics generation falls under the broad research area of language generation. Similar to traditional text generation tasks, it faces difficult challenges including maintaining grammatical accuracy and semantic consistency. In this project, we attempt to perform next-line lyrics generation using 1) simple N-gram model, 2) simple word level LSTM model, 3) concatenated word level (with character, phoneme, artists and genre representation) LSTM, and finally 4) fine-tuning GPT-2. We found that fine-tuning GPT-2 model provides us with the best results. During the process, we have also realized the difficulty in building and optimizing neural language models, as well as  the finicky and demanding nature of training LSTM model.


