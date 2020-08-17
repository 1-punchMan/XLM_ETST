OUTPATH=$PWD/data/processed/clts-en-zh/shared_emb


PRETRAINED=$PWD/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth

# DUMPED=$PWD/dumped/clts-ft-zhen/grmogu7fnf
# RELOAD_MODEL=$DUMPED/best-valid_zh-en_mt_bleu.pth
# RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name 'clts-ft-enzh' \
 --dump_path ./dumped \
 --reload_model "$PRETRAINED,$PRETRAINED" \
 --data_path $OUTPATH  \
 --lgs 'en-zh'  \
 --mt_steps 'en-zh'  \
 --encoder_only false \
 --emb_dim 512  \
 --n_layers 6  \
 --n_heads 8  \
 --dropout 0.1  \
 --attention_dropout 0.1  \
 --gelu_activation true  \
 --tokens_per_batch 3000 \
 --max_len 400  \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000  \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --eval_bleu true \
 --validation_metrics 'valid_en-zh_mt_bleu'  \
 --stopping_criterion 'valid_en-zh_mt_bleu',20  \
 --fp16 true \
 --amp 1 \
 --max_vocab 60000 \
 --share_encdec_emb true  \
 --share_inout_emb false \
 --accumulate_gradients 8 \
 --eval_rouge true
#  --reload_checkpoint "$RELOAD_CHECKPOINT"

