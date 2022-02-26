#
# Usage: train-clts-elmo.sh lg1-lg2
#

lgs=$1
OUTPATH=data/processed/clts-$lgs/word-char_60k
export CUDA_VISIBLE_DEVICES=1

# reload the pretrained XLM
PRETRAINED="/home/zchen/CLTS/dumped/xlm_en_zh/bptt512_max_len256/best-valid_en_mlm_ppl.pth"
MODEL="/home/zchen/CLTS/dumped/clts-elmo-en-zh/7fdu784vr1/best-valid_en-zh_mt_rouge1.pth"

CHECKPOINT="/home/zchen/CLTS/dumped/clts-elmo-en-zh/r5yxjj7jz5/checkpoint.pth"

python train.py \
    --exp_name clts-elmo-$lgs \
    --dump_path ./dumped \
    --reload_elmo "$MODEL" \
    --reload_model "$MODEL,$MODEL" \
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
    --use_xencoder true \
    --ts_emb_dim 512  \
    --ts_n_layers 6  \
    --ts_n_heads 8  \
    --ts_dropout 0.1  \
    --ts_attention_dropout 0.1  \
    --ts_gelu_activation true  \
    --bptt 512  \
    --max_len 512  \
    --max_epoch 100000 \
    --epoch_size 200000 \
    --share_inout_emb true \
    --eval_bleu true \
    --eval_rouge true \
    --validation_metrics valid_${lgs}_mt_rouge1  \
    --stopping_criterion valid_${lgs}_mt_rouge1,20  \
    --tokens_per_batch 3000 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003,warmup_updates=8000 \
    --xencoder_optimizer adam,lr=0.00002 \
    --label_smoothing 0.1 \
    --elmo_tune_lm true \
    --elmo_weights_dropout 0.0 \
    --elmo_final_dropout 0.2 \
    --elmo_layer_norm true \
    --elmo_affine_layer_norm true \
    --elmo_apply_softmax true \
    --elmo_channelwise_weights false \
    --elmo_scaled_sigmoid false \
    --elmo_individual_norms true \
    --elmo_channelwise_norm false \
    --elmo_init_gamma 1.0 \
    --elmo_ltn false \
    --elmo_ltn_dims "" \
    --elmo_train_gamma true \
    --fp16 true \
    --amp 1 \
    --accumulate_gradients 8 \
    \
    `# for evaluation` \
    --eval_only true \
    --eval_test_set true \
    --beam_size 4 \
    --length_penalty 0.8 \
    \
    # --reload_checkpoint $CHECKPOINT

