OUTPATH=$PWD/data/processed/clts-zh-en
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name 'clts_zhen-mlm_enzh' \
 --dump_path ./dumped \
 --reload_model 'mlm_enzh_512/checkpoint.pth,mlm_enzh_512/checkpoint.pth' \
 --data_path $OUTPATH  \
 --lgs 'zh-en'  \
 --mt_steps 'zh-en'  \
 --ts_emb_dim 512  \
 --ts_n_layers 6  \
 --ts_n_heads 8  \
 --ts_dropout 0.1  \
 --ts_attention_dropout 0.1  \
 --ts_gelu_activation true \
 --emb_dim 512  \
 --n_layers 6  \
 --n_heads 8  \
 --dropout 0.1  \
 --attention_dropout 0.1  \
 --gelu_activation true  \
 --batch_size 32  \
 --bptt 256   \
 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  \
 --epoch_size 300000   \
 --max_epoch 100000  \
 --validation_metrics 'valid_zh-en_mt_bleu'  \
 --stopping_criterion 'valid_zh-en_mt_bleu',10  
 --fp16 true \
 --amp 1
