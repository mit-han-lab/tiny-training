# our sparse update can achievehigher accuracy at lower memory usage
# here we use mcunet-5fps as an  example
# we use optimizer without momentum which saves memory (as in the main results of the paper)

# sparse update (100KB scheme): 88.84%
python train_cls.py   configs/transfer.yaml --run_dir runs/flowers/mcunet-5fps/sparse_100kb/sgd_qas_nomom \
    --net_name mcunet-5fps  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 22 --manual_weight_idx 21-24-27-30-36-39 \
    --weight_update_ratio 1-1-1-1-0.125-0.25

# update last 12 weights and biases (best setting for last k layers): 87.74%, uses 448KB
python train_cls.py   configs/transfer.yaml --run_dir runs/flowers/mcunet-5fps/full_update/sgd_qas_nomom \
    --net_name mcunet-5fps  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 12 --n_weight_update 12

# update full network: 1.09% (diverge, probably due to no BN)
python train_cls.py   configs/transfer.yaml --run_dir runs/flowers/mcunet-5fps/full_update/sgd_qas_nomom \
    --net_name mcunet-5fps  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom