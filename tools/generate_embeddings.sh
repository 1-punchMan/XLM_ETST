# OUTPATH=$PWD/data/processed/XLM_en_zh/50k/
OUTPATH=$PWD/data/processed/clts_xnli15-zh-en/shared_emb


# PRETRAINED=$PWD/dumped/xlm_mlm_enzh/93kejnkflp/best-valid_en_mlm_ppl.pth
PRETRAINED=$PWD/pretrained_models/mlm_xnli15_1024/mlm_xnli15_1024.pth


# DUMPED=$PWD/dumped/clts-ft-zhen/u0egqx9998
# RELOAD_MODEL=$DUMPED/best-valid_zh-en_mt_bleu.pth
# RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
CUDA_VISIBLE_DEVICES='0' python generate_embeddings.py --exp_name 'xlm_mlm_enzh' \
 --dump_path ./dumped \
 --reload_xencoder "$PRETRAINED" \
 --data_path $OUTPATH  \
 --lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'  \
 --clm_steps ''  \
 --mt_steps 'zh-en'  \
 --encoder_only false \
 --emb_dim 1024  \
 --n_layers 12  \
 --n_heads 16  \
 --dropout 0.1  \
 --attention_dropout 0.1  \
 --gelu_activation true  \
 --tokens_per_batch 3000 \
 --bptt 256  \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000  \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --validation_metrics _valid_en_mlm_ppl  \
 --stopping_criterion _valid_en_mlm_ppl,25  \
 --fp16 true \
 --amp 1 \
 --max_vocab 95000 \
 --share_encdec_emb true  \
 --share_inout_emb true \
 --gen_word_emb true
 #  --reload_checkpoint "$RELOAD_CHECKPOINT" \

