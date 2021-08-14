#
# Usage: train-clts-ft.sh lg1-lg2
#

lgs=$1
OUTPATH=data/processed/clts-$lgs/word-char_60k
export CUDA_VISIBLE_DEVICES=0

# reload the pretrained XLM
PRETRAINED="/home/zchen/CLTS/dumped/clts-ft-en-zh/a3wnk4ldrk/best-valid_en-zh_mt_rouge1.pth"

CHECKPOINT="/home/zchen/CLTS/dumped/clts-ft-en-zh/udaq91nglj/checkpoint.pth"

python train.py\
    --exp_name clts-ft-$lgs \
    --dump_path ./dumped \
    --reload_model "$PRETRAINED,$PRETRAINED" \
    `# --reload_checkpoint $CHECKPOINT` \
    --data_path $OUTPATH  \
    --lgs $lgs  \
    --mt_steps $lgs  \
    --encoder_only false \
    --emb_dim 512  \
    --n_layers 6  \
    --n_heads 8  \
    --dropout 0.1  \
    --attention_dropout 0.1  \
    --gelu_activation true  \
    --tokens_per_batch 3000 \
    --batch_size 32  \
    --bptt 512   \
    --max_len 512  \
    --max_epoch 100000  \
    --epoch_size 200000 \
    --eval_bleu true \
    --validation_metrics valid_${lgs}_mt_rouge1  \
    --stopping_criterion valid_${lgs}_mt_rouge1,10  \
    --share_encdec_emb true  \
    --share_inout_emb false \
    --eval_rouge true \
    --label_smoothing 0.0 \
    --optimizer adam,beta1=0.9,beta2=0.98,lr=0.00005\
    --eval_only true