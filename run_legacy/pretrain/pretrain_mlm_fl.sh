device_ids=$1
pretrain_source=$2
batch=$3
world=$4

CUDA_VISIBLE_DEVICES=$device_ids \
python3 ../../train.py \
--train_task pretrain \
--pretrain_src $pretrain_source \
--pretrain_task mlm \
--pred_model performer \
--structure fl \
--pred_pooling mean \
--batch_size $batch \
--world_size $world \
--criterion cross_entropy \
--wandb