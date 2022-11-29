# here we compare transfer learning accuracy w/ and w/o qas
# we update the last 6 layers to simulate partial update
# the learning rate here are obtained from grid search (i.e., the best learning rate)
# it changes for different models/datasets/update schemes
# we grid search the best learning rate for all {models x dataset x update schemes}

# w/o qas: 84.26%
python train_cls.py   configs/transfer.yaml --run_dir runs/flowers/mcunet-5fps/6b+6w/sgd \
    --net_name mcunet-5fps  --bs256_lr  0.1  --optimizer_name sgd \
    --enable_backward_config 1 --n_bias_update 6 --n_weight_update 6

# w/ qas: 88.08%
python train_cls.py   configs/transfer.yaml --run_dir runs/flowers/mcunet-5fps/6b+6w/sgd_qas \
    --net_name mcunet-5fps  --bs256_lr  0.075  --optimizer_name sgd_scale \
    --enable_backward_config 1 --n_bias_update 6 --n_weight_update 6

