OUTPATH=$PWD/data/processed/XLM_en_zh/50k
# export NGPU=2; python -m torch.distributed.launch --nproc_per_node=$NGPU
python train.py --exp_name 'xlm_mlm+tlm_enzh'  \
--dump_path ./dumped  \
--data_path $OUTPATH  \
--lgs 'en-zh'  \
--clm_steps '' \
--mlm_steps 'en,zh,en-zh'  \
--emb_dim 1024  \
--n_layers 12  \
--n_heads 16  \
--dropout 0.1  \
--attention_dropout 0.1  \
--gelu_activation true \
--batch_size 8  \
--bptt 256   \
--optimizer adam,lr=0.0001  \
--epoch_size 300000   \
--max_epoch 100000  \
--validation_metrics _valid_en_mlm_ppl  \
--stopping_criterion _valid_en_mlm_ppl,25 \
--fp16 false  
