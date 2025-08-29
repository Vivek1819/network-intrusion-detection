from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
