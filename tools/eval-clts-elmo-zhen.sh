OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb


# PRETRAINED=$PWD/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth

EXPNAME=sbcfjivn4w
DUMPED=$PWD/dumped/clts-elmo-zhen/$EXPNAME
RELOAD_MODEL=$DUMPED/best-valid_zh-en_mt_rouge1.pth
# RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-elmo-zhen' \
 --dump_path ./dumped \
 --reload_elmo "$RELOAD_MODEL" \
 --reload_model "$RELOAD_MODEL,$RELOAD_MODEL" \
 --data_path $OUTPATH  \
 --lgs 'zh-en'  \
 --mt_steps 'zh-en'  \
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
 --max_vocab 60000 \
 --tokens_per_batch 3000 \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005,warmup_updates=8000  \
 --xencoder_optimizer adam,lr=0.00001 \
 --accumulate_gradients 1 \
 --label_smoothing 0.1 \
 --elmo_tune_lm false \
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
 --eval_only true \
 --beam 4
#  --reload_checkpoint "$RELOAD_CHECKPOINT"

