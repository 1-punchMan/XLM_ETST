# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-wiki.sh lg1-lg2
#

set -e


# install tools
./install-tools.sh


#
# Initialize tools and data paths
#

# main paths
SRC=${1%-*}
TGT=${1#*-}
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
CLTS_PATH=$DATA_PATH/clts/$SRC-$TGT
PROC_PATH=$DATA_PATH/processed/clts-$SRC-$TGT/word-char_60k
VOC_PATH=$DATA_PATH/processed/XLM_en_zh/word-char_60k/vocab

# create paths
mkdir -p $PROC_PATH

CLEAR=$TOOLS_PATH/clear.py
TOKENIZE=$TOOLS_PATH/zh_char_tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py


# raw files
SRC_TRAIN_RAW=$CLTS_PATH/train-doc
TGT_TRAIN_RAW=$CLTS_PATH/train-sum

SRC_VALID_RAW=$CLTS_PATH/valid-doc
TGT_VALID_RAW=$CLTS_PATH/valid-sum

SRC_TEST_RAW=$CLTS_PATH/test-doc
TGT_TEST_RAW=$CLTS_PATH/test-sum


outerloop="train valid test"
 
########## tokenize ##########
innerloop="doc sum"
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


########## binarize ##########
innerloop="en zh"
for t in $outerloop; do
  for lg in $innerloop; do
    if [ ! -f $PROC_PATH/$t.en-zh.$lg.pth ]; then   
      if [ "$lg" == "$TGT" ]; then doc_sum="sum"; else doc_sum="doc"; fi

      TOK=$CLTS_PATH/$t-$doc_sum.tok
      echo "Binarizing $TOK data into $PROC_PATH/$t.en-zh.$lg.pth"

      ##### Use the different vocabulary lists to binarize src and tgt data  #####
      # if [ "$lg" == "zh" ]; then
      #   python $MAIN_PATH/preprocess.py $SRC_VOCAB $BPE
      # else
      #   python $MAIN_PATH/preprocess.py $TGT_VOCAB $BPE
      # fi
      ############################################################################
      
      ##### Use the shared vocabulary list to binarize src and tgt data  #####
      python preprocess.py $VOC_PATH $TOK
      ########################################################################

      mv $CLTS_PATH/$t-$doc_sum.pth $PROC_PATH/$t.en-zh.$lg.pth
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

