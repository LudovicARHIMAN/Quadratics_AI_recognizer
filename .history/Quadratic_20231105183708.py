import sympy as sp
import re

# Define a function to extract and solve quadratic coefficients
def extract_and_solve_coefficients(equation_str):
    # Define the symbols and the variable
    x, a, b, c = sp.symbols('x a b c')

    # Create a pattern to match the quadratic equation structure
    pattern = r"(\d*)\*x\*\*2 \+ (\d*)\*x \+ (\d*) = 0"

    match = re.match(pattern, equation_str)
    
    if match:
        a_val, b_val, c_val = map(int, match.groups())
        eq = sp.Eq(a * x**2 + b * x + c, 0)
        
        # Solve for the coefficients
        solutions = sp.solve(eq.subs({a: a_val, b: b_val, c: c_val}), x)
        return a_val, b_val, c_val, solutions

    return None

# Test the function with an example equation
example_eq = "1x^2  + 2*x + 3 = 0"
coefficients_and_solutions = extract_and_solve_coefficients(example_eq)

if coefficients_and_solutions:
    a_val, b_val, c_val, solutions = coefficients_and_solutions
    print(f"Coefficients: a={a_val}, b={b_val}, c={c_val}")
    print(f"Solutions: {solutions}")
else:
    print("Not a valid quadratic equation.")
