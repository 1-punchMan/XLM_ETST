#
# Usage: train-clts-ft.sh lg1-lg2
#

lgs=$1
OUTPATH="/home/zchen/CLTS/data/processed/clts-en-zh/word-char_60k/"
export CUDA_VISIBLE_DEVICES=1

# reload the pretrained XLM
PRETRAINED="/home/chiamin/python-projects/XLM/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth"
MODEL="/home/zchen/CLTS/dumped/clts-baseline-zh-en/u2w83gkzye/best-valid_zh-en_mt_rouge1.pth"

CHECKPOINT="/home/zchen/CLTS/dumped/clts-ft-en-zh/hblynfrnvi/checkpoint.pth"

python train.py \
    --exp_name clts-ft-$lgs \
    --dump_path ./dumped \
    `# --reload_model "$PRETRAINED,$PRETRAINED"` \
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
    --bptt 512   \
    --max_len 512  \
    --max_epoch 100000  \
    --epoch_size 200000 \
    --eval_bleu true \
    --validation_metrics valid_${lgs}_mt_rouge1  \
    --stopping_criterion valid_${lgs}_mt_rouge1,20  \
    --share_encdec_emb true  \
    --share_inout_emb false \
    --eval_rouge true \
    --label_smoothing 0.1 \
    --optimizer adam,beta1=0.9,beta2=0.98,lr=0.00005 \
    --fp16 true \
    --amp 1 \
    --accumulate_gradients 8 \
    --use_lang_emb true \
    \
    `# for evaluation \
    --eval_only true \
    # --eval_test_set true \
    # --beam_size 4 \
    # --length_penalty 0.8` \
    \
    --reload_checkpoint $CHECKPOINT