#!/bin/bash

DATA_FOLDER="/h/nng/data/sentiment/bias"

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

SPLITS="test"
GENRES="eec"
FILES="input0"
AUGS="orig"

for GENRE in $GENRES; do
  a_folder="$DATA_FOLDER/$GENRE"
  for AUG in $AUGS; do 
    for SPLIT in $SPLITS; do
      for file in $FILES; do
        if [ -f "${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.raw.$file" ] && [ ! -f "${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.$file" ]; then
          echo "BPE encoding $GENRE/$AUG/$SPLIT/$file"
          python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA_FOLDER/$GENRE/$AUG/$SPLIT.raw.$file" \
            --outputs "$DATA_FOLDER/$GENRE/$AUG/$SPLIT.$file" \
            --workers 12 \
            --keep-empty;
        fi
      done
      if [ ! -f "${DATA_FOLDER}/${GENRE}/$AUG/${SPLIT}.label" ]; then
        cp ${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.raw.label ${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.label
      fi
    done
    
    for file in $FILES; do
      if [ ! -d "${DATA_FOLDER}/${GENRE}/${AUG}/bin/$file" ]; then
        fairseq-preprocess \
          --only-source \
          --testpref "$DATA_FOLDER/$GENRE/$AUG/test.$file" \
          --destdir "$DATA_FOLDER/$GENRE/$AUG/bin/$file" \
          --workers 12 \
          --srcdict dict.txt;
      fi
    done
    if [ ! -d "${DATA_FOLDER}/${GENRE}/$AUG/bin/label" ]; then
      if [[ "$AUG" == *"soft"* ]] || [[ "$AUG" == *"ord"* ]]; then
        tname="soft"
        dname="soft"
      elif [[ "$AUG" == *"imp"* ]]; then
        tname="imp"
        dname="raw"
      else
        tname="raw"
        dname="raw"
      fi
      
      if [[ "$AUG" == *"reg"* ]]; then
        mkdir $DATA_FOLDER/$GENRE/$AUG/bin/label
        cp $DATA_FOLDER/$GENRE/$AUG/train.$tname.label $DATA_FOLDER/$GENRE/$AUG/bin/label/train.label
        cp $DATA_FOLDER/$GENRE/$AUG/valid.$dname.label $DATA_FOLDER/$GENRE/$AUG/bin/label/valid.label
      else
        fairseq-preprocess \
          --only-source \
          --testpref "$DATA_FOLDER/$GENRE/$AUG/test.$dname.label" \
          --destdir "$DATA_FOLDER/$GENRE/$AUG/bin/label" \
          --workers 12 \
          --srcdict "$DATA_FOLDER/label.dict.txt";
      fi
    fi
  done
done

