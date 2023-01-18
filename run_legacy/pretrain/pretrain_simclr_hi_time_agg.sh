CUDA_VISIBLE_DEVICES=3,4,5,7 \
python3 ../../train.py \
--input_path /home/edlab/jykim/UniHPF_pretrain/input \
--train_task pretrain \
--pretrain_src mimiciii \
--pretrain_task simclr \
--structure hi \
--pred_model eventaggregator \
--pred_pooling mean \
--n_layers 2 \
--embed_dim 128 \
--max_seq_len 256 \
--batch_size 16 \
--world_size 4 \
--criterion simclr \
--model unihpf_simclr \
--time_embed aggregator \
--wandb
