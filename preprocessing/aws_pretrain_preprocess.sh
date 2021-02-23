#!/bin/bash

DATA_FOLDER="/h/nng/data/sentiment/$1"

# download bpe encoder.json, vocabulary and fairseq dictionary
if [ ! -f "encoder.json" ]; then
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
fi
if [ ! -f "vocab.bpe" ]; then
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
fi
if [ ! -f "dict.txt" ]; then
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
fi

SPLITS="train valid test"
FILES="input0"
GENRE=$2
AUG=$3

a_folder="$DATA_FOLDER/$GENRE"
for SPLIT in $SPLITS; do
  python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs $a_folder/$AUG/$SPLIT.raw.input0 \
    --outputs $a_folder/$AUG/$SPLIT.bpe.input0 \
    --keep-empty \
    --workers 60
done

fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --trainpref $a_folder/$AUG/train.bpe.input0 \
    --validpref $a_folder/$AUG/valid.bpe.input0 \
    --testpref $a_folder/$AUG/test.bpe.input0 \
    --destdir $a_folder/$AUG/bin \
    --workers 60
