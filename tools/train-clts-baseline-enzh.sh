OUTPATH=$PWD/data/processed/clts-en-zh/shared_emb

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name 'clts-baseline-enzh' \
 --dump_path ./dumped \
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
 --batch_size 16 \
 --tokens_per_batch -1 \
 --max_len 400   \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005,warmup_updates=4000  \
 --max_epoch 100000  \
 --epoch_size 100000 \
 --eval_bleu true \
 --eval_rouge true \
 --validation_metrics 'valid_en-zh_mt_rouge1'  \
 --stopping_criterion 'valid_en-zh_mt_rouge1',20  \
 --fp16 true \
 --amp 1 \
 --accumulate_gradients 8 \
 --max_vocab 60000 \
 --share_encdec_emb true  \
 --share_inout_emb false
 
#  --src_max_vocab 25000 \
#  --tgt_max_vocab 30000
#  --reload_model 'dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/best-valid_zh-en_mt_bleu.pth,dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/best-valid_zh-en_mt_bleu.pth' \
#  --reload_checkpoint 'dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/checkpoint.pth'
# --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0005,warmup_updates=8000  \
