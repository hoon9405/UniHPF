device_num=$1
world_size=$2
train_src=$3
batch_size=$4
pred_task=$5
seed=$6
max_word_len=$6

OMP_NUM_THREADS=32 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../../main.py \
--train_task scratch \
--train_src $train_src \
--structure hi \
--emb_type textbase \
--feature whole \
--pred_model eventaggregator \
--n_layers 2 \
--max_word_len $max_word_len \
--batch_size $batch_size \
--world_size $world_size \
--criterion prediction \
--valid_subset valid,test \
--pred_task $pred_task \
--maximize_best_checkpoint_metric \
--seed $seed \
--wandb \
--pooled_eval \
