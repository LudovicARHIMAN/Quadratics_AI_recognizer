from random import randint
import re
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored
from math import sqrt

# Define functions for generating quadratic and non-quadratic equations
def quadratic_eq():
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    return f"{a}x^2 + {b}x + {c} = 0"

def non_quadratic_eq():
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    d = randint(-10, 10)
    return f"{a}x^3 + {b}x^2 + {c}x + {d} = 0"

# Generate equations
quadratic_equations = [quadratic_eq() for _ in range(100)]
non_quadratic_equations = [non_quadratic_eq() for _ in range(100)]

# Define a function to extract coefficients from the equation
def extract_features(equation):
    match = re.match(r"(-?\d*)x\^2\s*\+\s*(-?\d*)x\s*\+\s*(-?\d*)\s*=\s*0", equation)
    if match:
        return [int(match.group(1) or 1), int(match.group(2) or 0), int(match.group(3) or 0)]
    return None

# Convert equations into feature vectors and labels
X = []
y = []

for equation in quadratic_equations:
    features = extract_features(equation)
    if features:
        X.append(features)
        y.append(1)  # Label 1 for quadratic equations

for equation in non_quadratic_equations:
    features = extract_features(equation)
    if features:
        X.append(features)
        y.append(0)  # Label 0 for non-quadratic equations

# Create a decision tree classifier and train it on the data
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X, y)

# Define a function to classify an equation as quadratic or non-quadratic
def is_quadratic_eq(eq, model):
    features = extract_features(eq)
    if features:
        pred = model.predict([features])[0]
        return pred == 1
    return False

# Function to calculate roots of a quadratic equation
def calc_root(a, b, c):
    delta = (b**2) - 4 * a * c

    if delta > 0:
        root1 = (-b - sqrt(delta)) / (2 * a)
        root2 = (-b + sqrt(delta)) / (2 * a)
        return root1, root2, "are both solutions of the equation"

    if delta == 0:
        return (-b) / (2 * a), "is the unique solution of the equation"

    if delta < 0:
        return "This equation has no solution for x in the real set"

# Test the is_quadratic_eq function with an example equation
example_eq = "2x^2 + 44x + 0 = 0"
is_quadratic = is_quadratic_eq(example_eq, dt_classifier)
print(f"Is the equation quadratic? {is_quadratic}")

if is_quadratic:
    match = re.match(r"(-?\d*)x\^2\s*\+\s*(-?\d*)x\s*\+\s*(-?\d*)\s*=\s*0", example_eq)
    a = int(match.group(1) or 1)
    b = int(match.group(2) or 0)
    c = int(match.group(3) or 0)
    roots = calc_root(a, b, c)
    print(f"Coefficients: a={a}, b={b}, c={c}")
    print(f"Solutions: {roots}")
