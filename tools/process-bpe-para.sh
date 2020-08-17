# build the training set for BPE tokenization (50k codes)
pair="en-zh"
OUTPATH=$PWD/data/processed/clts-zh-en/shared_emb
FASTBPE=$PWD/tools/fastBPE/fast  # path to the fastBPE tool
FULL_VOCAB=$OUTPATH/vocab.$pair
EN_VOCAB=$OUTPATH/vocab.en
ZH_VOCAB=$OUTPATH/vocab.zh



# Build training set for the BPE vocabulary
mkdir -p $OUTPATH
shuf -r -n 5000000 data/wiki/txt/en.train >> $OUTPATH/bpe.train
shuf -r -n 5000000 data/wiki/txt/zh.train >> $OUTPATH/bpe.train


# Learn BPE vocabulary
$FASTBPE learnbpe 50000 $OUTPATH/bpe.train > $OUTPATH/codes




# Apply BPE
# echo "########## For parallel data... ##########"
# # Apply BPE tokenization on parallel train/valid/test files
# for lg in $(echo $pair | sed -e 's/\-/ /g'); do
#   for split in train valid test; do

#     echo "Apply BPE on: data/para/$pair.$lg.$split"
#     $FASTBPE applybpe $OUTPATH/$pair.$lg.$split data/para/$pair.$lg.$split $OUTPATH/codes
#     echo "Preprocess $OUTPATH/$pair.$lg.$split"
#     python preprocess.py $OUTPATH/vocab.$pair $OUTPATH/$pair.$lg.$split

#   done
# done

# # Get the post-BPE vocabulary
# cat $OUTPATH/en-zh.en.train >> $OUTPATH/en-zh.train
# cat $OUTPATH/en-zh.zh.train >> $OUTPATH/en-zh.train
# cat $OUTPATH/en-zh.train | $FASTBPE getvocab - > $OUTPATH/vocab

# # Binarize sentences
# for lg in $(echo $pair | sed -e 's/\-/ /g'); do
#   for split in train valid test; do
#     python preprocess.py $OUTPATH/vocab.$pair $OUTPATH/$pair.$lg.$split
#   done
# done



##############################################################################################
echo "########## For monolingual data ... ##########"
# Apply BPE tokenization on monolingual train/valid/test files
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    
    echo "Apply BPE to data/wiki/txt/$lg.$split"
    $FASTBPE applybpe $OUTPATH/$split.$lg data/wiki/txt/$lg.$split $OUTPATH/codes
    
    # echo "Preprocess $OUTPATH/$split.$lg"
    # python preprocess.py $OUTPATH/vocab.$pair $OUTPATH/$split.$lg
    
  done
done


# extract full vocabulary
echo "Extracting full vocabulary..."
$FASTBPE getvocab $OUTPATH/train.en $OUTPATH/train.zh > $FULL_VOCAB
echo "Full vocab in: $FULL_VOCAB"

# extract full vocabulary
echo "Extracting english vocabulary..."
$FASTBPE getvocab $OUTPATH/train.en > $EN_VOCAB
echo "English vocab in: $EN_VOCAB"

# extract full vocabulary
echo "Extracting chinese vocabulary..."
$FASTBPE getvocab $OUTPATH/train.zh > $ZH_VOCAB
echo "Chinese vocab in: $ZH_VOCAB"



# Binarize sentences
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    python preprocess.py $OUTPATH/vocab.$pair $OUTPATH/$split.$lg
  done
done
##############################################################################################


