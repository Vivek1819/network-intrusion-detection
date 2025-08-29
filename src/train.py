from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def train_mlp(X_train, y_train, hidden_layers=(128, 64), max_iter=50):
    """
    Trains an MLP Classifier (neural network).
    """
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp


def train_binary_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, num_classes):
    model = XGBClassifier(
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
    model.fit(X_train, y_train)
    return model