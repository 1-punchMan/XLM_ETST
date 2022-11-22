# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

pair=cn-tw
PARAPATH=data/cn-tw_1k/txt
VOC_PATH="/home/zchen/encyclopedia-text-style-transfer/data/vocab"
OUTPATH=data/cn-tw_1k/numericalized

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    for split in train valid test; do
        python preprocess.py $VOC_PATH $PARAPATH/$pair.$lg.$split
        mv $PARAPATH/$pair.$lg.$split.pth $OUTPATH/$split.$pair.$lg.pth
    done
done
