import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src import preprocess
import joblib

def test_data_shape():
    X_train, X_test, y_train, y_test = preprocess.load_and_preprocess()
    assert X_train.shape[1] == 4

def test_model_accuracy():
    model = joblib.load("src/model.pkl")
    _, X_test, _, y_test = preprocess.load_and_preprocess()
    acc = model.score(X_test, y_test)
    assert acc > 0.8
