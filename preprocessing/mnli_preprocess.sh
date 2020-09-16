#!/bin/bash

GLUE_DATA_FOLDER="/scratch/ssd001/datasets/nli/mnli"

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
GENRES="government telephone travel fiction slate"
FILES="input0 input1"
#BINS="metro_soft"
BINS="cbert"

for GENRE in $GENRES; do
  a_folder="$GLUE_DATA_FOLDER/$GENRE"
  for BIN in $BINS; do
    if [[ "$BIN" != *"orig"* ]] && [[ "$BIN" != *"eda"* ]]; then
      for file in $FILES; do 
        cat $a_folder/$BIN/train.gen.${file}_* $a_folder/orig/train.raw.$file > $a_folder/$BIN/train.raw.$file
      done
      cat $a_folder/$BIN/train.imp.label_* $a_folder/orig/train.raw.label > $a_folder/$BIN/train.imp.label
      cat $a_folder/$BIN/train.raw.label_* $a_folder/orig/train.raw.label > $a_folder/$BIN/train.raw.label
      cat $a_folder/$BIN/train.soft.label_* $a_folder/orig/train.soft.label > $a_folder/$BIN/train.soft.label
      cp $a_folder/orig/valid* $a_folder/$BIN
      cp $a_folder/orig/test* $a_folder/$BIN
    fi
    
    for SPLIT in $SPLITS; do
      for file in $FILES; do
        if [ -f "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/${SPLIT}.raw.$file" ] && [ ! -f "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/${SPLIT}.$file" ]; then
          echo "BPE encoding $GENRE/$SPLIT/$BIN/$file"
          python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs "$GLUE_DATA_FOLDER/$GENRE/$BIN/$SPLIT.raw.$file" \
            --outputs "$GLUE_DATA_FOLDER/$GENRE/$BIN/$SPLIT.$file" \
            --workers 12 \
            --keep-empty;
        fi
      done
    done
  
    if [[ "$GENRE" != "oup" && "$GENRE" != "nineeleven" && "$GENRE" != "letters" && "$GENRE" != "facetoface" && "$GENRE" != "verbatim" ]]; then
      for file in $FILES; do
        if [ ! -d "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/bin/$file" ]; then
          fairseq-preprocess \
            --only-source \
            --trainpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/train.$file" \
            --validpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/valid.$file" \
            --testpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/test.$file" \
            --destdir "$GLUE_DATA_FOLDER/$GENRE/$BIN/bin/$file" \
            --workers 12 \
            --srcdict dict.txt;
        fi
      done
      if [ ! -d "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/bin/label" ]; then
        if [[ "$BIN" == *"soft"* ]]; then
          tname="soft"
          dname="soft"
        elif [[ "$BIN" == *"imp"* ]]; then
          tname="imp"
          dname="raw"
        else
          tname="raw"
          dname="raw"
        fi
        fairseq-preprocess \
          --only-source \
          --trainpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/train.$tname.label" \
          --validpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/valid.$dname.label" \
          --destdir "$GLUE_DATA_FOLDER/$GENRE/$BIN/bin/label" \
          --workers 12 \
          --srcdict "$GLUE_DATA_FOLDER/label.dict.txt";
      fi
      cp $GLUE_DATA_FOLDER/$GENRE/$BIN/train.soft.label $GLUE_DATA_FOLDER/$GENRE/$BIN/bin/label/train.label
      cp $GLUE_DATA_FOLDER/$GENRE/orig/valid.soft.label $GLUE_DATA_FOLDER/$GENRE/$BIN/bin/label/valid.label
    else
      for file in $FILES; do
        if [ ! -d "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/bin/$file" ]; then
          fairseq-preprocess \
            --only-source \
            --trainpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/valid.$file" \
            --testpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/test.$file" \
            --destdir "$GLUE_DATA_FOLDER/$GENRE/$BIN/bin/$file" \
            --workers 12 \
            --srcdict dict.txt;
        fi
      done
      if [ ! -d "${GLUE_DATA_FOLDER}/${GENRE}/$BIN/bin/label" ]; then
        fairseq-preprocess \
          --only-source \
          --trainpref "$GLUE_DATA_FOLDER/$GENRE/$BIN/valid.raw.label" \
          --destdir "$GLUE_DATA_FOLDER/$GENRE/$BIN/bin/label" \
          --workers 12 \
          --srcdict "$GLUE_DATA_FOLDER/label.dict.txt";
      fi
    fi
  done
done
