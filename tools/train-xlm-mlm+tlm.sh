OUTPATH=data/processed/XLM_en_zh/word-char_60k
CHPTPATH="/home/zchen/CLTS/dumped/xlm_en_zh/2ziw0mieyg/checkpoint.pth"
export CUDA_VISIBLE_DEVICES=1
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
python train.py\
    --exp_name 'xlm_en_zh'  \
    --dump_path ./dumped  \
    --data_path $OUTPATH  \
    --lgs 'en-zh'  \
    --clm_steps '' \
    --mlm_steps 'en,zh,en-zh'  \
    --emb_dim 512  \
    --n_layers 6  \
    --n_heads 8  \
    --dropout 0.1  \
    --attention_dropout 0.1  \
    --gelu_activation true \
    --batch_size 32  \
    --bptt 512   \
    --max_len 256  \
    --optimizer adam,lr=0.0001  \
    --epoch_size 300000   \
    --max_epoch 100000  \
    --validation_metrics _valid_en_mlm_ppl  \
    --eval_only false\
    --stopping_criterion _valid_en_mlm_ppl,25
    # --reload_checkpoint $CHPTPATH