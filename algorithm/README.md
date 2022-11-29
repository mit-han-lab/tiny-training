# On-Device Training Under 256KB Memory

In this section, we provide the code to simulate the on-device training on GPU servers, including Quantization-Aware Scaling (QAS) and Sparse Update.

## Setups

**Environment setup.** We recommend using Anaconda to set up the environment. Please find an example set up below:

```bash
conda create -n mcunetv3 python=3.8
conda activate mcunetv3
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install easydict
pip install timm
```

**Dataset preparation**. 

1. We use ImageNet for pre-training and quantization calibration. We provided the quantized models so there is no need for the pre-training dataset. 
2. We benchmark the transfer learning performance on multiple downstream datasets. Please use the `make_all_datasets.sh` script [here](https://github.com/mit-han-lab/tinyml/tree/master/tinytl/dataset_setup_scripts) to fetch and process all downstream datasets. 
3. For VWW dataset, please follow the guide [here](https://github.com/tensorflow/models/tree/master/research/slim) to prepare the dataset. Alternatively, you may find our processed version here (**TODO**) (in `torchvision.datasets.ImageFolder` format). 

## Usage

### Model Quantization

We generate real quantized models from floating-point PyTorch models, which can simulate the quantized operators on MCUs. Please refer to `quantize/quantized_ops_diff.py` for details.

Here we have prepared three models in `int8` format: `mbv2-w0.35, mcunet-5fps, proxyless-w0.3`. After ImageNet pre-training, we perform post-training quantization (PTQ) and save it using a customized format under `assets/mcu_models`. We chose the models since they can fit the tight memory constraints. 

The ImageNet accuracies of the three models are:

| model          | accuracy (ptq) |
| -------------- | -------------- |
| mbv2-w0.35     | 45.7%          |
| mcunet-5fps    | 54.1%          |
| proxyless-w0.3 | 48.3%          |

### Quantization-Aware Scaling (QAS)

We can compare the performance with and without QAS. Please refer to `scripts/compare_qas.sh`.

Here `sgd` optimizer refers to training without QAS; `sgd_scale` refers to SGD with QAS. The accuracy of the Flowers-102 dataset is :

|         | accuracy |
| ------- | -------- |
| w/o QAS | 84.3%    |
| w/ QAS  | 88.1%    |

We can see that QAS significantly helps convergence. 

Note that we performed a grid search to find the optimal learning rate for each setting (different models/datasets/update schemes). You may need to tune the learning rate for other settings. 

### Sparse Update

Please find the script tp perform sparse update here: `scripts/sparse_update.sh`. 

We used the 100KB config of MCUNet as an example. We used the optimizers without momentum to save memory (same setting as the main results in the paper).

We can see that the sparse update scheme achieves 88.8% accuracy on Flowers-102, which is better than updating the whole model. For comparison, updating the last 12 layers leads to 87.7% accuracy while using 448KB memory.

