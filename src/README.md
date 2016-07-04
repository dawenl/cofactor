# Experimental Results
By getting the data and running the following notebooks, you should be able to reproduce the experimental results in the paper.

## ML20M
- [preprocess_ML20M.ipynb](./preprocess_ML20M.ipynb): pre-process the data and create the train/test/validation splits.
- [Cofactorization_ML20M.ipynb](./Cofactorization_ML20M.ipynb): train the CoFactor model and evaluate on the heldout test set.
- [WMF_ML20M.ipynb](./WMF_ML20M.ipynb): train the baseline WMF model and evaluate on the heldout test set.

To get the results for TasteProfile, follow [this notebook](https://github.com/dawenl/expo-mf/blob/master/src/processTasteProfile.ipynb) for data pre-processing and replace the data directory `DATA_DIR` in the above notebooks with the location of the processed TasteProfile.
