#!/bin/bash

DATA_FOLDER="/scratch/ssd001/datasets/nng_dataset/nli"

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

SPLITS="train valid test"
GENRES="anli mednli mnli snli"
FILES="input0 input1"
#AUGS="sampling_0.05_imp sampling_0.3_imp sampling_0.4_imp sampling_0.5_imp"
AUGS="orig"

for GENRE in $GENRES; do
  a_folder="$DATA_FOLDER/$GENRE"
  for AUG in $AUGS; do 
    if [[ "$AUG" != *"orig"* ]] && [[ "$AUG" != *"eda"* ]]; then
      for file in $FILES; do 
        cat $a_folder/$AUG/train.gen.${file}_* $a_folder/orig/train.raw.$file > $a_folder/$AUG/train.raw.$file
      done
      cat $a_folder/$AUG/train.imp.label_* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.imp.label
      cat $a_folder/$AUG/train.raw.label* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.raw.label
      cat $a_folder/$AUG/train.soft.label* $a_folder/orig/train.soft.label > $a_folder/$AUG/train.soft.label
      cp $a_folder/orig/valid* $a_folder/$AUG
      cp $a_folder/orig/test* $a_folder/$AUG
    fi

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
    done
    
    for file in $FILES; do
      if [ ! -d "${DATA_FOLDER}/${GENRE}/${AUG}/bin/$file" ]; then
        fairseq-preprocess \
          --only-source \
          --trainpref "$DATA_FOLDER/$GENRE/$AUG/train.$file" \
          --validpref "$DATA_FOLDER/$GENRE/$AUG/valid.$file" \
          --testpref "$DATA_FOLDER/$GENRE/$AUG/test.$file" \
          --destdir "$DATA_FOLDER/$GENRE/$AUG/bin/$file" \
          --workers 12 \
          --srcdict dict.txt;
      fi
    done
    if [ ! -d "${DATA_FOLDER}/${GENRE}/$AUG/bin/label" ]; then
      if [[ "$AUG" == *"soft"* ]]; then
        tname="soft"
        dname="soft"
      elif [[ "$AUG" == *"imp"* ]]; then
        tname="imp"
        dname="raw"
      else
        tname="raw"
        dname="raw"
      fi
      fairseq-preprocess \
        --only-source \
        --trainpref "$DATA_FOLDER/$GENRE/$AUG/train.$tname.label" \
        --validpref "$DATA_FOLDER/$GENRE/$AUG/valid.$dname.label" \
        --testpref "$DATA_FOLDER/$GENRE/$AUG/test.$dname.label" \
        --destdir "$DATA_FOLDER/$GENRE/$AUG/bin/label" \
        --workers 12 \
        --srcdict "$DATA_FOLDER/label.dict.txt";
    fi
    cp $GLUE_DATA_FOLDER/$GENRE/$AUG/train.soft.label $GLUE_DATA_FOLDER/$GENRE/$AUG/bin/label/train.label
    cp $GLUE_DATA_FOLDER/$GENRE/orig/valid.soft.label $GLUE_DATA_FOLDER/$GENRE/$AUG/bin/label/valid.label
  done
done
