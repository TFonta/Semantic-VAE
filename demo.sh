python demo.py --name RGB_model_no_bg --exclude_bg --load_size 256 \
    --model VAE \
     --sample_dir ./results/vae_kl_lin_dec_LSTM_weighted_bidir \
     --checkpoints_dir checkpoints/CA2SIS/VAE \
    --batchSize 2 --dataset_mode custom \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0 --no_flip --no_T --lstm_num 3 --bidir --CA2SIS \
     --cross_att_all_layers --single_layer_mask_enc --no_self_last_layers --multi_scale_style_enc --style_enc_feat_dim 4
    