from joblib import load
art = load("artifacts/baseline.joblib")
print("Scaler expects:", art["scaler"].n_features_in_)
