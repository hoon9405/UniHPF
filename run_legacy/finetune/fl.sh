device_ids=$1
pretrain_source=$2
pred_source=$3
batch=$4
world=$5

pretrain_methods="simclr mlm spanmlm"
tasks="mort los3 los7 readm dx"
for mth in $pretrain_methods
do
    for task in $tasks
    do
        CUDA_VISIBLE_DEVICES=$device_ids \
        python3 ../../train.py \
        --train_task finetune \
        --ft_pt_src $pretrain_source \
        --ft_pt_task $mth \
        --pred_src $pred_source \
        --structure fl \
        --pred_model performer \
        --max_seq_len 8192 \
        --pred_pooling mean \
        --batch_size $batch \
        --world_size $world \
        --criterion binary_cross_entropy \
        --valid_subset valid,test \
        --pred_target $task \
        --wandb \
        --best_checkpoint_metric auprc \
        --maximize_best_checkpoint_metric
    done
done

pretrain_methods="simclr mlm spanmlm"
tasks="fi_ac im_disch"
for mth in $pretrain_methods
do
    for task in $tasks
    do
        CUDA_VISIBLE_DEVICES=$device_ids \
        python3 ../../train.py \
        --train_task finetune \
        --ft_pt_src $pretrain_source \
        --ft_pt_task $mth \
        --pred_src $pred_source \
        --structure fl \
        --pred_model performer \
        --max_seq_len 8192 \
        --pred_pooling mean \
        --batch_size $batch \
        --world_size $world \
        --criterion cross_entropy \
        --valid_subset valid,test \
        --pred_target $task \
        --wandb \
        --best_checkpoint_metric auprc \
        --maximize_best_checkpoint_metric
    done
done



