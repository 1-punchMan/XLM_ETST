# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# set -e

CODES=50000 
N_THREADS=16    # number of threads in data preprocessing


RELOAD_CODES=data/processed/XLM_en_zh/50k/codes
RELOAD_VOCAB=data/processed/XLM_en_zh/50k/vocab.en-zh
# RELOAD_CODES=pretrained_models/mlm_xnli15_1024/codes_xnli_15
# RELOAD_VOCAB=pretrained_models/mlm_xnli15_1024/vocab_xnli_15

#
# Check parameters
#
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi


#
# Initialize tools and data paths
#

# main paths
SRC=zh
TGT=en
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
CLTS_PATH=$DATA_PATH/clts/$SRC-$TGT
PROC_PATH=$DATA_PATH/processed/clts-$SRC-$TGT/50k

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $CLTS_PATH
mkdir -p $PROC_PATH

CLEAR=$TOOLS_PATH/clear.py
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py


# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast


# raw files
SRC_TRAIN_RAW=$CLTS_PATH/train-doc
TGT_TRAIN_RAW=$CLTS_PATH/train-sum

SRC_VALID_RAW=$CLTS_PATH/valid-doc
TGT_VALID_RAW=$CLTS_PATH/valid-sum

SRC_TEST_RAW=$CLTS_PATH/test-doc
TGT_TEST_RAW=$CLTS_PATH/test-sum


# tokenized files
SRC_TRAIN_TOK=$SRC_TRAIN_RAW.tok
TGT_TRAIN_TOK=$TGT_TRAIN_RAW.tok

SRC_VALID_TOK=$SRC_VALID_RAW.tok
TGT_VALID_TOK=$TGT_VALID_RAW.tok

SRC_TEST_TOK=$SRC_TEST_RAW.tok
TGT_TEST_TOK=$TGT_TEST_RAW.tok


# BPE data
SRC_TRAIN_BPE=$PROC_PATH/train.en-zh.$SRC.bpe
TGT_TRAIN_BPE=$PROC_PATH/train.en-zh.$TGT.bpe

SRC_VALID_BPE=$PROC_PATH/valid-en-zh.$SRC.bpe
TGT_VALID_BPE=$PROC_PATH/valid-en-zh.$TGT.bpe
 
SRC_TEST_BPE=$PROC_PATH/test-en-zh.$SRC.bpe
TGT_TEST_BPE=$PROC_PATH/test-en-zh.$TGT.bpe


# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT


outerloop="train valid test"
innerloop="doc sum"
 
########## tokenize ##########
for t in $outerloop; do
  for doc_sum in $innerloop; do
    if [ "$doc_sum" == "doc" ]; then lg=$SRC; else lg=$TGT; fi
    
    if [ ! -f $CLTS_PATH/$t-$doc_sum.tok ]; then    
      RAW=$CLTS_PATH/$t-$doc_sum
      TOK=$CLTS_PATH/$t-$doc_sum.tok
      echo "Tokenize $RAW into $TOK ..."
      eval "cat $RAW | python $CLEAR | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $TOK"
    fi
  done
done
##############################


########## BPE ##########
# reload BPE codes
cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi
# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TRAIN_TOK $TGT_TRAIN_TOK > $BPE_CODES
  echo "BPE learned in $BPE_CODES"
fi

# apply BPE codes
innerloop="en zh"
for t in $outerloop; do
  for lg in $innerloop; do

    if [ ! -f $PROC_PATH/$t.en-zh.$lg.bpe ]; then    
      if [ "$lg" == "$TGT" ]; then doc_sum="sum"; else doc_sum="doc"; fi
      # if [ "$doc_sum" == "doc" ]; then lg=$SRC; else lg=$TGT; fi
      TOK=$CLTS_PATH/$t-$doc_sum.tok
      BPE=$PROC_PATH/$t.en-zh.$lg.bpe
      
      echo "Applying $TOK BPE codes to $BPE"
      $FASTBPE applybpe $BPE $TOK $BPE_CODES
    fi
  done
done
#############################################################


########## vocabulary ##########
# extract source and target vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TRAIN_BPE > $TGT_VOCAB
fi
echo "$SRC vocab in: $SRC_VOCAB"
echo "$TGT vocab in: $TGT_VOCAB"

# reload full vocabulary
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE $TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"
##############################


########## binarize ##########
innerloop="en zh"
for t in $outerloop; do
  for lg in $innerloop; do
    if [ ! -f $PROC_PATH/$t.en-zh.$lg.pth ]; then    
      BPE=$PROC_PATH/$t.en-zh.$lg.bpe      
      echo "Binarizing $BPE data into $PROC_PATH/$t.en-zh.$lg.pth"

      ##### Use the different vocabulary lists to binarize src and tgt data  #####
      # if [ "$lg" == "zh" ]; then
      #   python $MAIN_PATH/preprocess.py $SRC_VOCAB $BPE
      # else
      #   python $MAIN_PATH/preprocess.py $TGT_VOCAB $BPE
      # fi
      ############################################################################
      
      ##### Use the shared vocabulary list to binarize src and tgt data  #####
      python $MAIN_PATH/preprocess.py $FULL_VOCAB $BPE
      ########################################################################
    fi
  done
done
##############################


#
# Summary
#
echo ""
echo "===== Data summary"
echo "CLTS training data:"
echo "    $SRC: $PROC_PATH/train.en-zh.$SRC.pth"
echo "    $TGT: $PROC_PATH/train.en-zh.$TGT.pth"
echo "CLTS validation data:"
echo "    $SRC: $PROC_PATH/valid.en-zh.$SRC.pth"
echo "    $TGT: $PROC_PATH/valid.en-zh.$TGT.pth"
echo "CLTS testing data:"
echo "    $SRC: $PROC_PATH/test.en-zh.$SRC.pth"
echo "    $TGT: $PROC_PATH/test.en-zh.$TGT.pth"

