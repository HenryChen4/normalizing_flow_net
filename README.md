```
conda create -n "nfn_2d" python=3.10
conda activate "nfn_2d"
pip install -r requirements.txt
```

To run:

Train normalizing flow net on randomly sampled solutions.

```
python -m src.train_random
```

Train normalizing flow net on archive solutions.

```
python -m src.train_archive
```

Evaluate archive model.

```
python -m src.archive_model_eval
```

References:

- [IKFlow](https://arxiv.org/pdf/2111.08933)
- [Zuko](https://zuko.readthedocs.io/en/stable/index.html)
