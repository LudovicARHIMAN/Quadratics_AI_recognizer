from random import randint
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
import sklearn.linear_model as sk
from math import *




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



# Generate 100 quadratic and 100 non-quadratic equations
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create a decision tree classifier and train it on the training set
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def accuracy_color():
    if accuracy >= 1.0:
        return colored(accuracy, 'green')
    return colored(accuracy, 'red')



# check if an equation is quadratic or not from the dataset
def is_quadratic_eq(eq, model):
    features = extract_features(eq)
    if features:
        pred = model.predict([features])[0]
        if pred == 1:
            return colored(True, 'green')
        else:
            return colored(False, 'red')
    
    return colored(False, 'red')



# Solve the equation, find the root.s 
def calc_root(a,b,c):
    delta = (b**2)-4*a*c

    if delta >0:
        root1=(-(b)-sqrt(delta))/(2*a)
        root2=(-(b)+sqrt(delta))/(2*a)
        #root1 , root2 = str(root1), str(root2)
        
        return colored(root1,'red'), colored(root2)

    if delta==0:
        return (-(b))/(2*a) , "is the unique solution of the eqiation"

    if delta<0:
        return "This equation has no solution for x in the real set"


def solve_eq(eq): 
    match = re.match(r"(-?\d*)x\^2\s*\+\s*(-?\d*)x\s*\+\s*(-?\d*)\s*=\s*0", eq)
    if is_quadratic:
        a = int(match.group(1) or 1)
        b = int(match.group(2) or 0)
        c = int(match.group(3) or 0)

    return calc_root(a,b,c)
    


# Test the is_quadratic_eq function with an example equation
print("Accuracy:", accuracy_color())
example_eq = "2x^2 + 44x + 0 = 0"
is_quadratic = is_quadratic_eq(example_eq, dt_classifier)
print(f"Is the equation quadratic? {is_quadratic}")



print(solve_eq(example_eq))
