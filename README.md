# Search with dense vectors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RafaelGoncalves8/search-with-dense-vectors/blob/master/notebooks/Index.ipynb)
[![License](https://img.shields.io/github/license/RafaelGoncalves8/search-with-dense-vectors)](https://github.com/RafaelGoncalves8/search-with-dense-vectors/blob/master/LICENSE)

Final project for course on deep learning for nlp (IA376E/1s2020 @ Unicamp). This is an implementation of a Two Tower model for solving the problem of document retrieval (and passage ranking) in the dataset MSMarco. The project also uses queries generated using doc2query algotithm. The project is implemented using PyTorch and PyTorch Lighning, deep learning frameworks for Python.

## Docs (portuguese)

 The final article and the plan of work can be found in `docs/`.

## Usage

One can import the model in python or use as a script.

### Training

Example of training using model as module:

```python
from src.model import TwoTower
from pytorch_lightning import Trainer

model = TwoTower(**model_args)

trainer = Trainer(**trainer_args)
trainer.fit(model)
```

Example of training using train script:

```
   python src/train.ipy --gpus 1 --batch_size 32
 ```

There's also a colab notebook showing the usage in `notebooks/train.ipynb` and `notebooks/example.ipynb`.

## References

 - [doc2query](https://github.com/nyu-dl/dl4ir-doc2query)
 - [msmarco](https://microsoft.github.io/msmarco/)
 - [two tower model article](https://arxiv.org/abs/2002.03932)
 - [training model article](https://arxiv.org/abs/2004.04906)
