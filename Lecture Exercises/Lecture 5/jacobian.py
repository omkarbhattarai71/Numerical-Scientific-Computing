import numpy as np

# Black box function


def f(x):

    return np.log(np.dot(x.T,x)) * np.dot(np.outer(x,x),np.exp(x))


# Centroid approximation


def jacobian_centroid(x,eps):

    n = len(x)
    J = np.zeros((n,n))

    for i in range(n):

        p = np.zeros(n)
        p[i] = 1

        J[:,i] = (f(x+eps*p)-f(x-eps*p))/(2*eps)

    return J

# Complex step approximation

def jacobian_complex(x,eps):

    n = len(x)
    J = np.zeros((n,n))

    for i in range(n):

        p = np.zeros(n)
        p[i] = 1

        val = f(x + 1j*eps*p)

        J[:,i] = np.imag(val)/eps

    return J


# Main program

x0 = np.array([1,2,4,6,-2],dtype=float)

eps1 = 1e-8
eps2 = 1e-14


print("\nCentroid method (eps=1e-8)")
print(jacobian_centroid(x0,eps1))

print("\nCentroid method (eps=1e-14)")
print(jacobian_centroid(x0,eps2))


print("\nComplex step method (eps=1e-8)")
print(jacobian_complex(x0,eps1))

print("\nComplex step method (eps=1e-14)")
print(jacobian_complex(x0,eps2))