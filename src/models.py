from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def get_logistic_regression():
    return LogisticRegression(max_iter=1000, n_jobs=-1)

def get_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        max_iter=200,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True
    )
