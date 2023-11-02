from random import randint
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
from sklearn.linear_model import *
from math import *


def quadratic_eq():
    '''
    quadratics equations
    '''
    a = randint(-10,10)
    b = randint(-10,10)
    c = randint(-10,10)
    return f"{a}x^2 + {b}x + {c} = 0"



def non_quadratic_eq():
    '''
        non-quadratics equations
    '''
    a = randint(-10,10)
    b = randint(-10,10)
    c = randint(-10,10)
    d = randint(-10,10)
    return f"{a}x^3 + {b}x^2 + {c}x + {d} = 0"



# Generate 100 quadratics and non-quadratics equations
quadratic_equations = [quadratic_eq() for _ in range(100)]
non_quadratic_equations = [non_quadratic_eq() for _ in range(100)]



def extract_features(equation):
    ''' 
    take coefficient (a,b,c) from the equation 
    
    '''
   
    patt = re.match(r"(-?\d+)x\^2\s*\+\s*(-?\d+)x\s*\+\s*(-?\d+)\s*=\s*0", equation) # use regular espression to find if it's quadratics
    
    
    if patt:
        return [int(patt.group(1)), int(patt.group(2)), int(patt.group(3))] # give all the coeficient as a list
    
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



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier and train it on the training set
modl = DecisionTreeClassifier(random_state=42)
modl.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = modl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)



def accuracy_color():
    if accuracy >= 1.0:
        return colored(accuracy, 'green')
    else: 
        return colored(accuracy,'red')


def extract_features(equation):
    a , b , c = is_quadratic_eq(equation)
    if a != 0 : 
        return (a , b , c)
    else: 
        return None


def is_quadratic_eq(eq, model):
    features = extract_features(eq)
    if features:
        pred = model.predict([features])[0]
        if pred == 1:
            return colored(True,'green')
        else:
            return colored(False,'red')
    else:
        return colored(False,'red')



def CalcRacine(a,b,c):
    delta = (b**2)-4*a*c

    if delta >0:
        racine1=(-(b)-sqrt(delta))/(2*a)
        racine2=(-(b)+sqrt(delta))/(2*a)
        return racine1, racine2, "sont solution de l'equation"

    if delta==0:
        return (-(b))/(2*a) , "est la solution à l'équation"

    if delta<0:
        return "l'equation n'admet pas de solution"





# Train logistic regression (scikit-learn)
model = LogisticRegression()
model.fit(X, y)



print(accuracy_color())
eq = "x^2 + 5x + 2 = 0"
is_quadratic = is_quadratic_eq(eq , modl)


