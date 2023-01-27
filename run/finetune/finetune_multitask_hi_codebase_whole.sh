device_num=$1
world_size=$2
pt_src=$3
train_src=$4
batch_size=$5
max_word_len=$6

for seed in 42 43 44 45 46 
do
    OMP_NUM_THREADS=32 \
    CUDA_VISIBLE_DEVICES=$device_num \
    python3 ../../main.py \
    --train_task finetune \
    --pt_src $pt_src \
    --train_src $train_src \
    --structure hi \
    --emb_type codebase \
    --pretrain_task 'scratch' \
    --feature whole \
    --pred_model eventaggregator \
    --n_layers 2 \
    --max_word_len $max_word_len \
    --batch_size $batch_size \
    --world_size $world_size \
    --criterion prediction \
    --valid_subset valid,test \
    --maximize_best_checkpoint_metric \
    --seed $seed \
    --wandb 
done