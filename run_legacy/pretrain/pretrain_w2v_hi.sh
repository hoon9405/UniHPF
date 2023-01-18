device_ids=$1
pretrain_source=$2
batch=$3
world=$4

CUDA_VISIBLE_DEVICES=$device_ids \
python3 ../../train.py \
--train_task pretrain \
--pretrain_src $pretrain_source \
--pretrain_task w2v \
--structure hi \
--pred_model transformer \
--pred_pooling mean \
--n_layers 2 \
--embed_dim 128 \
--max_seq_len 256 \
--batch_size $batch \
--world_size $world \
--criterion w2v \
--wandb