import matplotlib
matplotlib.use('Agg')   

import numpy as np
import matplotlib.pyplot as plt

# Quadratic function

def f(x,a,b,c):
    return a*x**2 + b*x + c

# Quadratic root formulas

def quadratic_roots(a,b,c,dtype):

    a = dtype(a)
    b = dtype(b)
    c = dtype(c)

    alpha = np.sqrt(b*b - dtype(4)*a*c)

    xa1 = (-b + alpha)/(dtype(2)*a)
    xa2 = (-b - alpha)/(dtype(2)*a)

    xb1 = (dtype(2)*c)/(-b - alpha)
    xb2 = (dtype(2)*c)/(-b + alpha)

    return xa1,xa2,xb1,xb2


# PART A

print("\nPART A")

a = 1e-5
b = 1e3
c = 1e3

types = [np.float16,np.float32,np.float64]

for t in types:

    xa1,xa2,xb1,xb2 = quadratic_roots(a,b,c,t)

    print("\nFloating type:",t)
    print("xa1 =",xa1)
    print("xa2 =",xa2)
    print("xb1 =",xb1)
    print("xb2 =",xb2)


# PART B

print("\nPART B")

def condition_number(x0,xdelta,a,b,c):

    num = abs(f(x0+xdelta,a,b,c)-f(x0,a,b,c))
    den = abs(f(x0,a,b,c))

    return (num/den)*(abs(x0)/abs(xdelta))


xdelta = 1e-5
xhat = -1

d_values = [1e1,1e0,1e-1,1e-2,1e-3,1e-4]

for d in d_values:

    x = np.linspace(xhat-d,xhat+d,1000)
    kappa = condition_number(x,xdelta,a,b,c)

    plt.figure()
    plt.plot(x,kappa)
    plt.title("Condition number (d0="+str(d)+")")
    plt.xlabel("x0")
    plt.ylabel("kappa(x0)")
    plt.grid(True)

    filename = f"condition_partB_d{d}.png"
    plt.savefig(filename)
    plt.close()

    print("Saved:", filename)



# PART C


print("\nPART C")

x0 = -1.0
kappa = condition_number(x0,xdelta,a,b,c)

print("Condition number at x0=-1 :",kappa)



# PART D


print("\nPART D")

a = 1.0
b = -12.0
c = 20.0

xa1,xa2,xb1,xb2 = quadratic_roots(a,b,c,np.float64)

print("xa1 =",xa1)
print("xa2 =",xa2)
print("xb1 =",xb1)
print("xb2 =",xb2)

xhat = 2.000001

for d in d_values:

    x = np.linspace(xhat-d,xhat+d,1000)

    kappa = condition_number(x,xdelta,a,b,c)

    plt.figure()
    plt.plot(x,kappa)
    plt.title("Condition number plot around x=2 (d0="+str(d)+")")
    plt.xlabel("x0")
    plt.ylabel("kappa(x0)")
    plt.grid(True)

    filename = f"condition_partD_d{d}.png"
    plt.savefig(filename)
    plt.close()

    print("Saved:", filename)

