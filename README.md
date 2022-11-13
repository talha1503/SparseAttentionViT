# SparseAttentionViT


## Dataset

[Imagenett2](https://github.com/fastai/imagenette)

Due to GPU and data constraints

## Experiment

Metric: Classfication Accuracy
Loss: CrossEntropy
Optimizer: Adam

### Configurations

Original ViT:
ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)



## Results

| Model             | Accuracy    | 
| -----------       | ----------- |
| Original ViT      |    65.97    |
| Adapted BigBird           | -           |
| Random Attention           | -           |
| Windowed Attention           | -           |
| Global Attention           | -           |


Combinations of the above

