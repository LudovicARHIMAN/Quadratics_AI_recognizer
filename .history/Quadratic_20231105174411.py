import sympy as sp
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from termcolor import colored


# Define functions for generating quadratic and non-quadratic equations
def quadratic_eq():
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)
    eq = sp.Eq(a*sp.symbols('x')**2 + b*sp.symbols('x') + c, 0)
    return str(eq)

def non_quadratic_eq():
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)
    d = random.randint(1, 10)
    eq = sp.Eq(a*sp.symbols('x')**3 + b*sp.symbols('x')**2 + c*sp.symbols('x') + d, 0)
    return str(eq)

# Generate 100 quadratic and 100 non-quadratic equations
quadratic_equations = [quadratic_eq() for _ in range(100)]
non_quadratic_equations = [non_quadratic_eq() for _ in range(100)]

# Define a function to extract coefficients from the equation and solve it
def extract_coefficients_and_solve(equation):
    x = sp.symbols('x')
    match = re.match(r"(\d*)\*x\*\*2 \+ (\d*)\*x \+ (\d*) = 0", equation)
    if match:
        a, b, c = map(int, match.groups())
        eq = sp.Eq(a*x**2 + b*x + c, 0)
        solutions = sp.solve(eq, x)
        return a, b, c, solutions
    return None

# Convert equations into feature vectors and labels
X = []
y = []

for equation in quadratic_equations:
    coefficients_and_solutions = extract_coefficients_and_solve(equation)
    if coefficients_and_solutions:
        X.append(coefficients_and_solutions[:3])  # Coefficients
        y.append(1)  # Label 1 for quadratic equations

for equation in non_quadratic_equations:
    coefficients_and_solutions = extract_coefficients_and_solve(equation)
    if coefficients_and_solutions:
        X.append(coefficients_and_solutions[:3])  # Coefficients
        y.append(0)  # Label 0 for non-quadratic equations

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model and train it on the training set
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = logistic_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def accuracy_color():
    if accuracy >= 1.0:
        return colored(accuracy, 'green')
    return colored(accuracy, 'red')

print("Logistic Regression Classifier Accuracy:", accuracy_color())

# Define a function to check if an equation is quadratic
def is_quadratic_eq(eq, model):
    coefficients_and_solutions = extract_coefficients_and_solve(eq)
    if coefficients_and_solutions:
        pred = model.predict([coefficients_and_solutions[:3]])[0]
        if pred == 1:
            return colored(True, 'green'), coefficients_and_solutions[3]  # Solutions
        else:
            return colored(False, 'red'), None
    
    return colored(False, 'red'), None


def CalcRacine(a,b,c):
    delta = (b**2)-4*a*c

    if delta >0:
        root1=(-(b)-sqrt(delta))/(2*a)
        root2=(-(b)+sqrt(delta))/(2*a)
        return root1, root2, "are both solutions of the equation"

    if delta==0:
        return (-(b))/(2*a) , "is the unique solution of the eqiation"

    if delta<0:
        return "This equation has no solution for x in the real set"


# Test the is_quadratic_eq function with an example equation
example_eq = "1*x**2 + 2*x + 3 = 0"
is_quadratic, solutions = is_quadratic_eq(example_eq, logistic_classifier)
print(f"Is the equation quadratic? {is_quadratic}")
if solutions:
    print("Solutions:", solutions)


