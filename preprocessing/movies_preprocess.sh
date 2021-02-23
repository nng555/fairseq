#!/bin/bash

DATA_FOLDER="/h/nng/data/sentiment/movies"

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
GENRES="imdb"
#GENRES="imdb"
FILES="input0"
AUGS="sampling_0.1_soft sampling_0.2_soft sampling_0.3_soft sampling_0.4_soft sampling_0.5_soft sampling_0.6_soft"
#AUGS="books"
#AUGS="orig orig_0.01 orig_0.05 orig_0.1 orig_0.2 orig_0.3 orig_0.4 orig_0.5 orig_0.6 orig_0.7 orig_0.8 orig_0.9"
#AUGS="sampling_acc_0.51_soft sampling_acc_0.55_soft sampling_acc_0.61_soft sampling_acc_0.68_soft sampling_acc_0.71_soft sampling_acc_0.76_soft sampling_acc_0.78_soft sampling_acc_0.81_soft sampling_acc_0.51_imp sampling_acc_0.55_imp sampling_acc_0.61_imp sampling_acc_0.68_imp sampling_acc_0.71_imp sampling_acc_0.76_imp sampling_acc_0.78_imp sampling_acc_0.81_imp sampling_acc_0.51_hard sampling_acc_0.55_hard sampling_acc_0.61_hard sampling_acc_0.68_hard sampling_acc_0.71_hard sampling_acc_0.76_hard sampling_acc_0.78_hard sampling_acc_0.81_hard"
#$AUGS="orig_acc_0.51 orig_acc_0.55 orig_acc_0.61 orig_acc_0.68 orig_acc_0.71 orig_acc_0.76 orig_acc_0.78 orig_acc_0.81"
#AUGS="sampling_acc_0.5_soft sampling_acc_0.55_soft sampling_acc_0.6_soft sampling_acc_0.65_soft sampling_acc_0.7_soft sampling_acc_0.75_soft sampling_acc_0.8_soft sampling_acc_0.85_soft sampling_acc_0.9_soft sampling_acc_0.93_soft sampling_acc_0.5_hard sampling_acc_0.55_hard sampling_acc_0.6_hard sampling_acc_0.65_hard sampling_acc_0.7_hard sampling_acc_0.75_hard sampling_acc_0.8_hard sampling_acc_0.85_hard sampling_acc_0.9_hard sampling_acc_0.93_hard sampling_acc_0.5_imp"

for GENRE in $GENRES; do
  a_folder="$DATA_FOLDER/$GENRE"
  for AUG in $AUGS; do 
    if [[ "$AUG" != *"orig"* ]] && [[ "$AUG" != *"eda"* ]]; then
      for file in $FILES; do 
        cat $a_folder/$AUG/train.gen.${file}_* $a_folder/orig/train.raw.$file > $a_folder/$AUG/train.raw.$file
      done
      cat $a_folder/$AUG/train.imp.label_* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.imp.label
      cat $a_folder/$AUG/train.raw.label_* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.raw.label
      cat $a_folder/$AUG/train.soft.label_* $a_folder/orig/train.soft.label > $a_folder/$AUG/train.soft.label
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
      if [ ! -f "${DATA_FOLDER}/${GENRE}/$AUG/${SPLIT}.label" ]; then
        cp ${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.raw.label ${DATA_FOLDER}/${GENRE}/${AUG}/${SPLIT}.label
      fi
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
          --trainpref "$DATA_FOLDER/$GENRE/$AUG/train.$tname.label" \
          --validpref "$DATA_FOLDER/$GENRE/$AUG/valid.$dname.label" \
          --testpref "$DATA_FOLDER/$GENRE/$AUG/test.$dname.label" \
          --destdir "$DATA_FOLDER/$GENRE/$AUG/bin/label" \
          --workers 12 \
          --srcdict "$DATA_FOLDER/label.dict.txt";
        cp $DATA_FOLDER/$GENRE/$AUG/train.soft.label $DATA_FOLDER/$GENRE/$AUG/bin/label/train.label
        cp $DATA_FOLDER/$GENRE/orig/valid.soft.label $DATA_FOLDER/$GENRE/$AUG/bin/label/valid.label
      fi
    fi
  done
done

