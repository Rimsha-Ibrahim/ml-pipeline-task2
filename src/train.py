from preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train, X_test, y_train, y_test = load_and_preprocess()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

joblib.dump(model, "src/model.pkl")
