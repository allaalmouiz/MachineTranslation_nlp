# English to French Neural Machine Translation

## Overview

A Seq2Seq LSTM model that translates English sentences into French, built with TensorFlow/Keras. The model encodes an English sentence into a context vector and decodes it into French using teacher forcing during training and autoregressive decoding at inference.


## Dataset

137,860 English-French sentence pairs. 80/20 train/test split. Max English length: 15 tokens, max French length: 23 tokens. No null values.


## Architecture

| Component | Configuration |
|---|---|
| Encoder | Embedding (256) → LSTM (256 units) → state_h, state_c |
| Decoder | Embedding (256) → LSTM (256, return_sequences) → Dense (softmax) |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Parameters | 1,278,809 (~4.88 MB) |


## Training

12 epochs, batch size 64. Final train accuracy: 99.88%, val accuracy: 99.76%. Note: token-level accuracy is inflated by padding tokens and teacher forcing — BLEU score is a more reliable measure.


## Sample Translations

| English | Predicted French | Result |
|---|---|---|
| china is usually busy during september... | chine est généralement occupé en septembre... | Good |
| I like mangoes | jaime les mangues | Good |
| my favourite fruit is apple | mon fruit préféré est la pomme | Excellent |
| California is cold | ne vont ils à la californie | Poor |


## Improvements

Bidirectional LSTM encoder, attention mechanism, dropout, scheduled sampling, FastText pretrained embeddings (EN + FR), BLEU score evaluation, and subword tokenization (BPE) for OOV handling.

