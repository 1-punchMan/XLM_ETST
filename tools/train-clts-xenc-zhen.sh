OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb
# OUTPATH=$PWD/data/processed/clts_xnli15-zh-en/shared_emb


PRETRAINED=$PWD/dumped/xlm_mlm_enzh/ecd8gu1a43/best-valid_en_mlm_ppl.pth
# PRETRAINED=$PWD/pretrained_models/mlm_xnli15_1024/mlm_xnli15_1024.pth

#### if you want to resume a training process, define DUMPED, PRETRAINED, RELOAD_CHECKPOINT and uncomment the last 2 lines in training script
# DUMPED=$PWD/dumped/clts-xenc-zhen/9bhrodp5t8
# PRETRAINED=$DUMPED/best-valid_zh-en_mt_rouge1.pth
# RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth

# ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-xenc-zhen' \
 --dump_path ./dumped \
 --data_path $OUTPATH  \
 --reload_xencoder "$PRETRAINED" \
 --lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  \
 --mt_steps 'zh-en'  \
 --encoder_only false \
 --emb_dim 1024  \
 --n_layers 12  \
 --n_heads 16  \
 --dropout 0.1  \
 --attention_dropout 0.1  \
 --gelu_activation true  \
 --use_xencoder true \
 --ts_emb_dim 512  \
 --ts_n_layers 6  \
 --ts_n_heads 8  \
 --ts_dropout 0.1  \
 --ts_attention_dropout 0.1  \
 --ts_gelu_activation true  \
 --bptt 256  \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --share_inout_emb true \
 --eval_bleu true \
 --eval_rouge true \
 --validation_metrics 'valid_zh-en_mt_rouge1'  \
 --stopping_criterion 'valid_zh-en_mt_rouge1',20  \
 --fp16 true \
 --amp 1 \
 --max_vocab 95000 \
 --tokens_per_batch 1000 \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003,warmup_updates=12000 \
 --xencoder_optimizer adam,lr=0.00001 \
 --accumulate_gradients 20 \
 --label_smoothing 0.0
#  --reload_checkpoint "$RELOAD_CHECKPOINT" \
#  --reload_model "$PRETRAINED,$PRETRAINED"


