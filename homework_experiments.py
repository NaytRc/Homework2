import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# 3.1 
def run_hyperparameter_search(configs, train_fn):
    results = []
    for cfg in configs:
        # cfg = {lr, batch_size, optimizer}
        acc = train_fn(**cfg)
        results.append({**cfg, 'accuracy': acc})
    return results

# 3.2 
def add_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

if __name__ == '__main__':
    pass