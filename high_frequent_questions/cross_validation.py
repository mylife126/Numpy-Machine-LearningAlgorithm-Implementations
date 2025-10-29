"""
🧩 Cross Validation (CV) 核心逻辑总结

CV 的核心逻辑是在 (X_train, y_train) 上把数据划分为 K 个 folds，
在 K 次循环中，每次都使用不同的 fold 作为 validation set，
剩下的 folds 合并为 training set。这样模型在每次训练时
都能在不同的数据分布下被评估。

💡 目的：
通过多次训练–验证，可以更稳定地估计模型的
generalization ability（泛化能力），
而不是依赖于某一次随机划分可能不均衡的结果。

📈 原理上：
每一次 fold 代表模型在不同数据上的表现。
- 有的 fold 可能 fit 不足（bias 较高）
- 有的 fold 可能 fit 过度（variance 较高）
把它们的平均结果作为最终估计，
可以更全面地反映模型的整体 bias–variance balance。

⚙️ 因此，CV 常与 Grid Search 一起使用：
- 外层 Grid Search：循环不同的模型参数（hyperparameters）
- 内层 Cross Validation：评估当前参数下模型的平均验证性能

最终选择在所有参数组合中，CV 平均表现最好的那组超参数，
既能平衡 bias 与 variance，又能最大化泛化性能。
"""

from regression_glm import *
from sklearn.datasets import make_classification
from itertools import product


def get_k_fold(X_train, k, random_state):
    """
    1. Get the total numbers of X, m
    2. Randomize the index
    3. split the index into k folds
    list<list<indexs>>
    4. for each k fold, iterate K time, each loop get 1 fold to be validation set, and the rest is the training
    5. return list<tuple(train, val)>
    """

    m = X_train.shape[0]
    np.random.seed(random_state)

    m_index = list(range(m))
    np.random.shuffle(m_index)

    # now split the index into folds using np.array_split
    folds = np.array_split(m_index, k)

    splits = []
    for i in range(k):
        this_val_index = folds[i]
        this_train_index = np.setdiff1d(m_index, this_val_index)

        splits.append((this_train_index, this_val_index))

    return splits


def k_fold_cross_validation(X, y, k, random_state, evaluation_metrics, model, **paremeters):
    """
    The input is the X_train actually, because you must leave the test set to be independent
    """
    splits = get_k_fold(X, k, random_state)

    performances = []
    for split in splits:
        train_index, val_index = split
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        this_model = model(X_train, y_train, **paremeters)
        this_model.fit()
        y_pred = this_model.predict(X_val)

        performance = evaluation_metrics(y_val, y_pred)
        performances.append(performance)
    return np.mean(performances), np.std(performances)


def grid_search(X, y, k, random_state, evaluation_metrics, model, parameter_grid):
    """
    paremeter grid is dict(list(parameters))
    """
    all_parameter_keys = list(parameter_grid.keys())  # list of parameter names

    # gives you all the combinations a by b, suppose only 2 paremeters, and each length is a and b, then a by b combinations
    all_combinations = list(product(*parameter_grid.values()))

    best_performance = 0  # if our evaluation is accuracy
    best_setup = None
    for combination in all_combinations:
        the_paremeter_dict = dict(zip(all_parameter_keys, combination))  # since the dim of key and value is the same
        the_paremeter_dict["which_regression"] = "logistic"
        this_set_up_performance_via_kfold = k_fold_cross_validation(X, y, k, random_state, evaluation_metrics, model,
                                                                    **the_paremeter_dict)
        print(this_set_up_performance_via_kfold)

        if float(this_set_up_performance_via_kfold[0]) > best_performance:
            best_performance = this_set_up_performance_via_kfold[0]
            best_setup = the_paremeter_dict

    return best_performance, best_setup


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0)
    parameters = {"reg_lambda": 0.5, "lr": 0.01, "which_regression": "logistic"}
    # model = RegressionGLM(X, y, **parameters)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    this_set_up_performance_via_kfold = k_fold_cross_validation(X_train, y_train, 5,
                                                                random_state=42, evaluation_metrics=accuracy,
                                                                model=RegressionGLM,
                                                                **parameters)

    parameter_grid = {"lr": [0.1, 0.01, 0.001], "reg_lambda": [0.1, 0.5, 1]}

    best_performance = grid_search(X_train, y_train, 5,
                                   random_state=42, evaluation_metrics=accuracy,
                                   model=RegressionGLM,
                                   parameter_grid=parameter_grid)
