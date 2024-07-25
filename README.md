# MultiMax
This is the official implementation of our ICML 2024 paper "MultiMax: Sparse and Multi-Modal Attention Learning".

## Illustration of improved multi-modality and sparsity for 3-dimensional inputs
<p align="center">
   <img src="simplex_total.png" alt="drawing" width="600"/>
</p>
<p align="left">
   <b>Figure 1:</b> We evaluate SoftMax, SparseMax, EntMax, EvSoftMax and MultiMax (using the parameters of a hidden layer MultiMax trained on ImageNet directly) functions on a series of example input points $v ∈ R^3$ and project the resulting distribution on a simplex $∆^2$. Informally, the interior of the simplex stands for trimodal distributions, the edges constitute the set of bimodal distributions, and the vertices are unimodal distributions. Notably, the above figures highlight the advantage of MultiMax’s multi-modality. EntMax, Sparsemax and SoftMax with small temperature (blue colored line) yield a (quasi) uni-modal distribution, which ignore the second largest entry. In contrary, SoftMax with higher temperatures (green and orange colored line) fails to ignore the negative entry.
</p>


## Implementation
1. Efficient implementation:
- We implement the Max operator with 0 in Equation 6 as Pytorch built-in ReLU function
- We apply torch.jit.script decorator to fuse the remaining elementwise operations in Equation 6, following the [official documentation of TorchScript](https://pytorch.org/docs/stable/generated/torch.jit.script.html)
- We term the implementation of our modulation function as **Segmented Rectified Linear Unit (SeLU)**

2. Key changes in `vision_transformer.py`:
- The **modulator function** in Equation 6 of our paper is implemented in line 101.
- The **attention layer with MultiMax** is implemented at line 133 by modulating the input to SoftMax via SeLU.
- The **output layer with MultiMax** is implemented at line 324 in the same way.
- We adopt **Global Average Pooling (GAP)** instead of Classification Token to aggregate the spatial information for our baseline model.

## Experiments
### Train a Vision Transformer with MultiMax
1. Replace [timm/models/vision_transformer.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) with our provided `vision_transformer.py`
2. Folow the training procedure of [Deit](https://github.com/facebookresearch/deit) to reproduce our experiment results 

## Acknowledgements

This repo is based on [Deit](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models).

Thanks to the original authors for their great work!

## References

```bibtex

@InProceedings{pmlr-v235-zhou24g,
  title = 	 {{M}ulti{M}ax: Sparse and Multi-Modal Attention Learning},
  author =       {Zhou, Yuxuan and Fritz, Mario and Keuper, Margret},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {61897--61912},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/zhou24g.html},
}


