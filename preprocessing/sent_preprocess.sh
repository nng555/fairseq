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
#GENRES="books clothing home kindle movies pets sports tech tools toys men women baby shoes"
#GENRES="kindle clothing"
#GENRES="men women baby shoes"
FILES="input0"
#AUGS="sampling_0.03_imp sampling_0.05_imp sampling_0.07_imp sampling_0.09_imp sampling_0.11_imp sampling_0.13_imp sampling_0.17_imp sampling_0.19_imp sampling_0.21_imp sampling_0.23_imp"
#AUGS="sampling_0.15_imp"
#AUGS="sampling_ord"
#AUGS="sampling_0.15_soft sampling_0.3_soft sampling_0.4_soft sampling_0.5_soft sampling_0.15_imp sampling_0.3_imp sampling_0.4_imp sampling_0.5_imp"
#AUGS="sampling_0.15_soft_conv sampling_0.3_soft_conv sampling_0.4_soft_conv sampling_0.5_soft_conv"
#AUGS="eda"
#AUGS="uda"
GENRE=$2
AUG=$3

a_folder="$DATA_FOLDER/$GENRE"
if [[ "$AUG" != *"orig"* ]] && [[ "$AUG" == *"frac"* ]]; then
  frac=$(echo $AUG | perl -nle 'm/frac_([0-9]*)/; print $1')
  echo $frac
  for file in $FILES; do 
    cat $a_folder/$AUG/train.gen.${file}_* $a_folder/orig_frac_${frac}/train.raw.$file > $a_folder/$AUG/train.raw.$file
  done
  cat $a_folder/$AUG/train.imp.label_* $a_folder/orig_frac_${frac}/train.raw.label > $a_folder/$AUG/train.imp.label
  cat $a_folder/$AUG/train.raw.label_* $a_folder/orig_frac_${frac}/train.raw.label > $a_folder/$AUG/train.raw.label
  cat $a_folder/$AUG/train.soft.label_* $a_folder/orig_frac_${frac}/train.soft.label > $a_folder/$AUG/train.soft.label
  cp $a_folder/orig_frac_${frac}/valid* $a_folder/$AUG
  cp $a_folder/orig_frac_${frac}/test* $a_folder/$AUG
elif [[ "$AUG" != *"orig"* ]] && [[ "$AUG" != *"eda"* ]]; then
  for file in $FILES; do 
    cat $a_folder/$AUG/train.gen.${file}_* $a_folder/orig/train.raw.$file > $a_folder/$AUG/train.raw.$file
  done
  cat $a_folder/$AUG/train.imp.label_* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.imp.label
  cat $a_folder/$AUG/train.raw.label_* $a_folder/orig/train.raw.label > $a_folder/$AUG/train.raw.label
  cat $a_folder/$AUG/train.soft.label_* $a_folder/orig/train.soft.label > $a_folder/$AUG/train.soft.label
  cp $a_folder/orig/valid* $a_folder/$AUG
  cp $a_folder/orig/test* $a_folder/$AUG
fi
cp $a_folder/orig/valid* $a_folder/$AUG
cp $a_folder/orig/test* $a_folder/$AUG

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

