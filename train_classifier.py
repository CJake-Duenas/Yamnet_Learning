import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def main(npz="embeddings.npz", out_model="cry_clf.joblib"):
    data = np.load(npz)
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]

    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    joblib.dump({"scaler": scaler, "clf": clf}, out_model)
    print("Saved model to", out_model)

if __name__ == "__main__":
    main()
