# `ml/` — Classic Machine Learning

From-scratch NumPy + PyTorch implementations of the classic ML toolbox. See
[`../MAP.md`](../MAP.md) for the full catalogue and status.

| Sub-folder | Algorithms |
|---|---|
| `linear-models/` | linear regression (OLS/Ridge/Lasso/ElasticNet), logistic regression (binary/softmax) |
| `perceptron/` | Rosenblatt perceptron (+ averaged, pocket, multiclass) |
| `svm/` | SVM (linear & kernel), SVR |
| `knn/` | k-nearest neighbours (classification + regression) |
| `naive-bayes/` | Gaussian / Multinomial / Bernoulli NB |
| `trees/` | decision tree (CART), random forest, gradient boosting, AdaBoost |
| `ensemble/` | bagging, stacking, voting |
| `clustering/` | k-means, GMM (EM), DBSCAN, hierarchical, mean-shift, spectral |
| `dimensionality-reduction/` | PCA, LDA, t-SNE, UMAP, SVD, ICA |

Run any module directly, e.g. `python ml/clustering/kmeans.py`.
