OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb
DUMPED=$PWD/dumped/clts-ft-zhen/q5j5n3bbds
PRETRAINED=$DUMPED/best-valid_zh-en_mt_bleu.pth


CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-ft-zhen' \
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
 --bptt 256  \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000  \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --eval_bleu true \
 --validation_metrics 'valid_zh-en_mt_bleu'  \
 --stopping_criterion 'valid_zh-en_mt_bleu',20  \
 --fp16 true \
 --amp 1 \
 --max_vocab 60000 \
 --share_encdec_emb false  \
 --share_inout_emb true \
 --accumulate_gradients 8 \
 --eval_rouge true \
 --beam_size 4 \
 --eval_only true \
 --batch_size 32

