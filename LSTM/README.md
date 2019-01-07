# LSTM

## Predicter VAE-LSTM Many-2-One
Predicts an 8th sequence based on 7 input sequences

## Preprocessing of MIDI files

## 0. Go to the Preprocessors folder

## 1. If you have not done this for the autoencoder already, transpose your MIDI files to as many pitches as you would like to play in (max. 60 ~ 5 octaves)
```bash
python Transposer.py --file_path /path/to/dir --tp_by 30 --tp_step_size 1
```
You will end up with lots of midi files which TpBy** endings, which stands for transposed by.

## 2. 