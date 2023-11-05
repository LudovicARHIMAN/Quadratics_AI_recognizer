from random import randint
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
from sklearn.linear_model import LogisticRegression
from math import *
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split




def quadratic_eq():
    '''
    quadratics equations
    '''
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    return f"{a}x^2 + {b}x + {c} = 0"


def non_quadratic_eq():
    '''
        non-quadratics equations
    '''
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    d = randint(-10, 10)
    return f"{a}x^3 + {b}x^2 + {c}x + {d} = 0"


# Generate 100 quadratics and non-quadratics equations
quadratic_equations = [quadratic_eq() for _ in range(100)]
non_quadratic_equations = [non_quadratic_eq() for _ in range(200)]


def extract_features(equation):
    patt = re.match(r"(-?\d+)x\^2\s*\+\s*(-?\d+)x\s*\+\s*(-?\d+)\s*=\s*0", equation) 
    
    if match:
        a = int(match.group(1))
        return a, 0, 0  # Assuming that only the quadratic term is considered
    return None


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

for i in range(10):  # Print the first 10 equations and their labels
    print(f"Equation: {X[i]} | Label: {y[i]}")



#  Train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)


def is_quadratic_eq(eq):
    features = extract_features(eq)
    if features:
        pred = model.predict([features])[0]
        return pred == 1
    return False


def CalcRacine(a, b, c):
    delta = (b ** 2) - 4 * a * c

    if delta > 0:
        racine1 = (-(b) - sqrt(delta)) / (2 * a)
        racine2 = (-(b) + sqrt(delta)) / (2 * a)
        return racine1, racine2, "sont solution de l'equation"

    if delta == 0:
        return (-(b)) / (2 * a), "est la solution à l'équation"

    if delta < 0:
        return "l'equation n'admet pas de solution"
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Example usage
eq = "x^2 + 5x + 2 = 0"
is_quadratic = is_quadratic_eq(eq)
print(f"The equation '{eq}' is quadratic: {is_quadratic}")

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_test_pred))


class_counts = Counter(y)
print(class_counts)
