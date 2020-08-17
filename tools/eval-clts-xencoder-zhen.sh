OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb


# PRETRAINED=$PWD/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth

DUMPED=$PWD/dumped/clts-xencoder-zhen/o9343eg6yo
RELOAD_MODEL=$DUMPED/best-valid_zh-en_mt_rouge1.pth
# RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-xencoder-zhen' \
 --dump_path ./dumped \
 --reload_xencoder "$RELOAD_MODEL" \
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
 --tokens_per_batch 3000 \
 --bptt 256  \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0007,warmup_updates=4000  \
 --xencoder_optimizer adam,lr=0.00001 \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --eval_bleu true \
 --validation_metrics 'valid_zh-en_mt_rouge1'  \
 --stopping_criterion 'valid_zh-en_mt_rouge1',20  \
 --fp16 true \
 --amp 1 \
 --max_vocab 60000 \
 --share_encdec_emb true  \
 --share_inout_emb true \
 --accumulate_gradients 8 \
 --eval_rouge true \
 --eval_only true \
 --beam 4
#  --reload_checkpoint "$RELOAD_CHECKPOINT"