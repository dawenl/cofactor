# CoFactor

This repository contains the source code to reproduce the experimental results as described in the paper ["Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence"](http://dawenl.github.io/publications/LiangACB16-cofactor.pdf) (RecSys'16).

## Dependencies
The python module dependencies are:
- numpy/scipy
- scikit.learn
- joblib
- bottleneck
- pandas (needed to run the example for data preprocessing)

**Note**: The code is mostly written for Python 2.7. For Python 3.x, it is still usable with minor modification. If you run into any problem with Python 3.x, feel free to contact me and I will try to get back to you with a helpful solution.  

## Datasets
- [Taste Profile Subset](http://labrosa.ee.columbia.edu/millionsong/tasteprofile): the pre-processing is done following [this notebook](https://github.com/dawenl/expo-mf/blob/master/src/processTasteProfile.ipynb).
- [MovieLens-20M](http://grouplens.org/datasets/movielens/20m/)

We adapted the weighted matrix factorization (WMF) implementation from [content_wmf](https://github.com/dawenl/content_wmf) repository. 

## Examples
See example notebooks in `src/`. 
