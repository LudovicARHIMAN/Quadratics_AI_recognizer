from random import randint
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
from sklearn.linear_model import LogisticRegression
from math import *


# ... (The rest of your code remains the same)


# Convert the quadratic and non-quadratic equations into feature vectors (from regular expression)
X = []
y = []

for equation in quadratic_equations:
    features = extract_features(equation)
    if features:
        X.append(features)
        y.append(1)

for equation in non_quadratic_equations:
    features = extract_features(equation)
    if features:
        X.append(features)
        y.append(0)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# ... (The rest of your code)

print(accuracy_color())
eq = "x^2 + 5x + 2 = 0"
is_quadratic = is_quadratic_eq(eq, modl)
