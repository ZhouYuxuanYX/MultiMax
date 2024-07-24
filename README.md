# MultiMax
This is the official implementation of our ICML 2024 paper "MultiMax: Sparse and Multi-Modal Attention Learning""

# Usage


# Acknowledgements

This repo is based on [Deit](https://github.com/facebookresearch/deit). 

Thanks to the original authors for their work!

# References

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
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhou24g/zhou24g.pdf},
  url = 	 {https://proceedings.mlr.press/v235/zhou24g.html},
  abstract = 	 {SoftMax is a ubiquitous ingredient of modern machine learning algorithms. It maps an input vector onto a probability simplex and reweights the input by concentrating the probability mass at large entries. Yet, as a smooth approximation to the Argmax function, a significant amount of probability mass is distributed to other, residual entries, leading to poor interpretability and noise. Although sparsity can be achieved by a family of SoftMax variants, they often require an alternative loss function and do not preserve multimodality. We show that this trade-off between multi-modality and sparsity limits the expressivity of SoftMax as well as its variants. We provide a solution to this tension between objectives by proposing a piece-wise differentiable function, termed MultiMax, which adaptively modulates the output distribution according to input entry range. Through comprehensive analysis and evaluation, we show that MultiMax successfully produces a distribution that supresses irrelevant entries while preserving multi-modality, with benefits in image classification, language modeling and machine translation.}
}


