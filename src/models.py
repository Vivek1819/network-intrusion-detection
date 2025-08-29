from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_logistic_regression():
    return LogisticRegression(max_iter=1000, n_jobs=-1)

def get_mlp(hidden_layers=(128, 64), max_iter=200):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=42,
        activation="relu",
        solver="adam",
        early_stopping=True
    )

def get_binary_logreg():
    return LogisticRegression(max_iter=1000, n_jobs=-1)

def get_xgboost(num_classes):
    return XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
