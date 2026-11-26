
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Date features
dates = pd.to_datetime([
    "2026-01-10", "2026-02-15",
    "2026-03-20", "2026-04-12"
])

date_features = pd.DataFrame({
    "day": dates.day,
    "month": dates.month,
    "weekday": dates.dayofweek
})

print("Date Features:")
print(date_features)

# Ratio features
X["alcohol_ratio"] = X["alcohol"] / X["malic_acid"]
X["ash_ratio"] = X["ash"] / X["magnesium"]

# TF-IDF
notes = [
    "fruity floral",
    "fruity oak",
    "oak spicy"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(notes)

print("\nTF-IDF Features:")
print(vectorizer.get_feature_names_out())
print(tfidf.toarray())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Filter method
filter_model = SelectKBest(f_classif, k=5)
filter_model.fit(X_train, y_train)

# Wrapper method
rfe = RFE(
    LogisticRegression(max_iter=5000),
    n_features_to_select=5
)
rfe.fit(X_train, y_train)

# Embedded method - L1
l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=5000
)
l1.fit(X_train, y_train)

# Decision Tree importance
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Compare methods
methods = {
    "Filter": filter_model.transform,
    "Wrapper": rfe.transform
}

for name, transform in methods.items():
    Xtr = transform(X_train)
    Xte = transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)

    print("\n", name)
    print("Features:", Xtr.shape[1])
    print("Accuracy:", accuracy_score(y_test, pred))

# L1 selected features
selected = np.any(l1.coef_ != 0, axis=0)

Xtr = X_train.loc[:, selected]
Xte = X_test.loc[:, selected]

model = LogisticRegression(max_iter=5000)
model.fit(Xtr, y_train)

print("\nL1")
print("Features:", Xtr.shape[1])
print("Accuracy:", accuracy_score(
    y_test, model.predict(Xte)
))

# Decision Tree selected features
indices = np.argsort(tree.feature_importances_)[-5:]

Xtr = X_train.iloc[:, indices]
Xte = X_test.iloc[:, indices]

model = LogisticRegression(max_iter=5000)
model.fit(Xtr, y_train)

print("\nDecision Tree")
print("Features:", Xtr.shape[1])
print("Accuracy:", accuracy_score(
    y_test, model.predict(Xte)
))
