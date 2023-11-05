from random import randint
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter


def quadratic_eq():
    '''
    Quadratic equations
    '''
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    return f"{a}x^2 + {b}x + {c} = 0"


def non_quadratic_eq():
    '''
    Non-quadratic equations
    '''
    a = randint(-10, 10)
    b = randint(-10, 10)
    c = randint(-10, 10)
    d = randint(-10, 10)
    return f"{a}x^3 + {b}x^2 + {c}x + {d} = 0"


# Generate 100 quadratic and 100 non-quadratic equations
quadratic_equations = [quadratic_eq() for _ in range(100)]
non_quadratic_equations = [non_quadratic_eq() for _ in range(100)]


def extract_features(equation):
    patt = re.match(r"(-?\d+)x\^2\s*\+\s*(-?\d+)x\s*\+\s*(-?\d+)\s*=\s*0", equation)

    if patt:
        return [int(patt.group(1)), int(patt.group(2)), int(patt.group(3))]
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




# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_test_pred)
print(report)

# Count the class distribution
class_counts = Counter(y)
print(class_counts)

