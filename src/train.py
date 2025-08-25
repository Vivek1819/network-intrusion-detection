from preprocessing import load_and_preprocess
from models import get_logistic_regression, get_mlp
from evaluate import evaluate_model

if __name__ == "__main__":
    file_path = "../data/UNSW_NB15_training-set.csv"

    # Preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess(file_path)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = get_logistic_regression()
    log_reg.fit(X_train, y_train)
    evaluate_model(log_reg, X_test, y_test)

    # MLP
    print("\nTraining MLP Classifier...")
    mlp = get_mlp()
    mlp.fit(X_train, y_train)
    evaluate_model(mlp, X_test, y_test)
