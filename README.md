# search-with-dense-vectors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RafaelGoncalves8/search-with-dense-vectors/blob/master/notebooks/Index.ipynb)
[![License](https://img.shields.io/github/license/RafaelGoncalves8/search-with-dense-vectors)](https://github.com/RafaelGoncalves8/search-with-dense-vectors/blob/master/LICENSE)

Final project for course on deep learning for nlp (IA376E/1s2020 @ Unicamp). This is an implementation of a Two Tower model for solving the problem of document retrieval (and passage ranking) in the dataset MSMarco. The project also uses queries generated using doc2query algotithm. The project is implemented using PyTorch and PyTorch Lighning, deep learning frameworks for Python.

## Abstract

## Usage

One can import the model in python:

### Training

```python
from src.model import TwoTower
from pytorch_lightning import Trainer

model = TwoTower(**model_args)

trainer = Trainer(**trainer_args)
trainer.fit(model)
```

## References
