OUTPATH=$PWD/data/processed/XLM_en_zh/50k

DUMPED=dumped/xlm_mlm_enzh/vbqnrflui4
RELOAD_MODEL=$DUMPED/best-valid_en_mlm_ppl.pth
RELOAD_CHECKPOINT=$DUMPED/checkpoint.pth

CUDA_VISIBLE_DEVICES=1 python train.py \
 --exp_name 'xlm_mlm_enzh'  \
 --dump_path ./dumped \
 --data_path $OUTPATH  \
 --lgs 'en-zh'  \
 --clm_steps ''  \
 --mlm_steps 'en,zh'  \
 --emb_dim 512  \
 --n_layers 6  \
 --n_heads 8  \
 --dropout 0.1  \
 --attention_dropout 0.1  \
 --gelu_activation true  \
 --batch_size 32  \
 --bptt 256   \
 --optimizer adam,lr=0.0001  \
 --epoch_size 300000   \
 --max_epoch 98  \
 --validation_metrics _valid_en_mlm_ppl  \
 --stopping_criterion _valid_en_mlm_ppl,25  \
 --fp16 true \
 --amp 1 \
 --accumulate_gradients 4 \
 --max_vocab 60000 \
 --reload_checkpoint "$RELOAD_CHECKPOINT" \
 --reload_model "$RELOAD_MODEL"