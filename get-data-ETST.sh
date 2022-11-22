# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-para.sh $lg_pair
#

set -e

pair=$1  # input language pair

# data paths
MAIN_PATH=$PWD
PARA_PATH=$PWD/data/baidu-wiki/txt

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/zh_char_tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py

# install tools
./install-tools.sh

#
# Tokenize and preprocess data
#

# tokenize
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    cat $PARA_PATH/$pair.$lg.${split}_raw | $TOKENIZE zh | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.$split
  done
done