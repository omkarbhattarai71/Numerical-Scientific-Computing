# Exercise 8.9: Deliberately poorly-written module for pylint demonstration
# This code has many issues that pylint will detect and report

import math

# Non-descriptive variable names
x = 5
y = 10
z = 0

# Function without docstring
def solve_quad(a,b,c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    else:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return x1, x2


# Unused variable
unused_constant = 42

# Dead code
def old_function():
    pass


# Poor code style: inconsistent spacing and naming
def compute( data):
    result=sum(data)/len(data)
    return result


# Global variable used inconsistently
GLOBAL_FLAG=True

# Complex function without clear purpose
def process_data(input_list, flag=False):
    output = []
    if flag:
        for element in input_list:
            if element > 0:
                output.append(element * 2)
            else:
                output.append(element)
    else:
        output = input_list
    return output


# Function with too many branches (high complexity)
def categorize_value(x):
    if x < -10:
        return "very_negative"
    elif x < -5:
        return "negative"
    elif x < 0:
        return "slightly_negative"
    elif x == 0:
        return "zero"
    elif x < 5:
        return "slightly_positive"
    elif x < 10:
        return "positive"
    else:
        return "very_positive"


# Bare except clause (bad practice)
def risky_operation():
    try:
        result = 1 / 0
    except:
        print("Error occurred")


# Missing docstring for class
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value = self.value + x
    
    def subtract(self, x):
        self.value = self.value-x
    
    def get_value(self):
        return self.value


# Inconsistent indentation and weird spacing
def messy_function():
    x=1
    y=   2
    z=3
    return x+y+z


# Line too long (over 100 characters)
def long_line_function():
    very_long_variable_name_that_makes_this_line_exceed_the_typical_100_character_limit_significantly = "This is a very long string that demonstrates poor code formatting practices"
    return very_long_variable_name_that_makes_this_line_exceed_the_typical_100_character_limit_significantly


if __name__ == '__main__':
    result = solve_quad(1, -3, 2)
    print(result)
