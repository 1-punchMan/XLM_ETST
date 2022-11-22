#
# Usage: train-clts-ft.sh lg1-lg2
#

lgs=$1
DATAPATH="/home/zchen/XLM_ETST/data/cn-tw_1k/numericalized/"
export CUDA_VISIBLE_DEVICES=0

# reload the pretrained XLM
PRETRAINED="/home/chiamin/python-projects/XLM/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth"

CHECKPOINT="/home/zchen/XLM_ETST/dumped/PST_cn-tw/av7mf60q1q/checkpoint.pth"

python train.py \
    --exp_name PST_${lgs}_1k \
    --dump_path ./dumped \
    `# --reload_model "$PRETRAINED,$PRETRAINED"` \
    --data_path $DATAPATH  \
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
    --bptt 256   \
    --max_len 256  \
    --max_epoch 100000  \
    --epoch_size 17384 \
    --eval_bleu true \
    --validation_metrics valid_${lgs}_mt_bleu \
    --stopping_criterion valid_${lgs}_mt_bleu,15 \
    --share_encdec_emb true  \
    --share_inout_emb false \
    `# --label_smoothing 0.1` \
    --optimizer adam,beta1=0.9,beta2=0.98,lr=0.00001 \
    --fp16 true \
    --amp 1 \
    `# --accumulate_gradients 8` \
    --use_lang_emb false \
    \
    `# for evaluation \
    --eval_only true \
    # --eval_test_set true \
    # --beam_size 4` \
    --length_penalty 0.8 \
    \
    # --reload_checkpoint $CHECKPOINT