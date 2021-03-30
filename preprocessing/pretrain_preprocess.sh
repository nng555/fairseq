#!/bin/bash

DATA_FOLDER="/h/nng/data/pretrain/$1"

# download bpe encoder.json, vocabulary and fairseq dictionary
#if [ ! -f "/h/nng/projects/robust_nli/preprocessing/encoder.json" ]; then
#  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
#fi
#if [ ! -f "vocab.bpe" ]; then
#  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
#fi
#if [ ! -f "dict.txt" ]; then
#  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
#fi

SPLITS="train valid test"
FILES="input0"
GENRE=$2

a_folder="$DATA_FOLDER/$GENRE"
for SPLIT in $SPLITS; do
  python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json /h/nng/projects/robust_nli/preprocessing/encoder.json \
    --vocab-bpe /h/nng/projects/robust_nli/preprocessing/vocab.bpe \
    --inputs $a_folder/$SPLIT.raw.input0 \
    --outputs $a_folder/$SPLIT.bpe.input0 \
    --keep-empty \
    --workers 30
done

fairseq-preprocess \
    --only-source \
    --srcdict /h/nng/projects/robust_nli/preprocessing/dict.txt \
    --trainpref $a_folder/train.bpe.input0 \
    --validpref $a_folder/valid.bpe.input0 \
    --testpref $a_folder/test.bpe.input0 \
    --destdir $a_folder/bin \
    --workers 30
