OUTPATH=$PWD/data/processed/clts-zh-en/clean_shared_emb

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name 'clts-baseline-zhen' \
 --dump_path ./dumped \
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
 --max_len 100   \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003,warmup_updates=4000  \
 --max_epoch 100000  \
 --epoch_size 200000 \
 --eval_bleu true \
 --eval_rouge true \
 --validation_metrics 'valid_zh-en_mt_rouge1'  \
 --stopping_criterion 'valid_zh-en_mt_rouge1',20  \
 --fp16 true \
 --amp 1 \
 --accumulate_gradients 2 \
 --max_vocab 60000 \
 --share_encdec_emb false  \
 --share_inout_emb true \
 --label_smoothing 0.0
 
#  --src_max_vocab 25000 \
#  --tgt_max_vocab 30000
#  --reload_model 'dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/best-valid_zh-en_mt_bleu.pth,dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/best-valid_zh-en_mt_bleu.pth' \
#  --reload_checkpoint 'dumped/clts_zhen-mlm_enzh-scratch/dyy2ekhp64/checkpoint.pth'
