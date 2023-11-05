import sympy as sp
import random
import re
from sklearn.linear_model import LogisticRegression

# Define functions for generating quadratic and non-quadratic equations
def quadratic_eq():
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)
    eq = sp.Eq(a*sp.symbols('x')**2 + b*sp.symbols('x') + c, 0)
    return str(eq)

# Generate quadratic equations
quadratic_equations = [quadratic_eq() for _ in range(100)]

# Define a function to extract coefficients from the equation
def extract_coefficients(equation):
    x = sp.symbols('x')
    match = re.match(r"(\d*)\*x\*\*2 \+ (\d*)\*x \+ (\d*) = 0", equation)
    if match:
        a, b, c = map(int, match.groups())
        return a, b, c
    return None

# Convert equations into feature vectors (coefficients)
X = []
y = []

for equation in quadratic_equations:
    coefficients = extract_coefficients(equation)
    if coefficients:
        X.append(coefficients)
        y.append(coefficients)  # Use coefficients as target for training

# Create a logistic regression model and train it on the coefficients
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X, y)

# Define a function to predict coefficients from an equation
def predict_coefficients(eq, model):
    coefficients = extract_coefficients(eq)
    if coefficients:
        predicted_coefficients = model.predict([coefficients])
        return predicted_coefficients[0]
    return None

# Test the predict_coefficients function with an example equation
example_eq = "1*x**2 + 2*x + 3 = 0"
predicted_coefficients = predict_coefficients(example_eq, logistic_classifier)
print(f"Predicted Coefficients: {predicted_coefficients}")
