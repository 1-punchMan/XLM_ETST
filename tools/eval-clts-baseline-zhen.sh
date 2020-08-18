OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb
DUMPED=$PWD/dumped/clts-baseline-zhen/o397vtnn3h
PRETRAINED=$DUMPED/best-valid_zh-en_mt_rouge1.pth

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-baseline-zhen' \
 --dump_path ./dumped \
 --reload_model "$PRETRAINED,$PRETRAINED" \
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
 --share_encdec_emb false  \
 --share_inout_emb true \
 --accumulate_gradients 8 \
 --eval_rouge true \
 --eval_only true \
 --beam 4

