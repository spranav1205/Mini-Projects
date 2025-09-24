import numpy as np
import matplotlib.pyplot as plt

def bisection(f, l, r, t = 1e-10, max_iterations = 100):

    if (f(l) * f(r) > 0):
        raise ValueError("No Root in the given interval")

    for i in range(max_iterations):
        m = l + (r-l)/2
        if ((f(m) >= 0 and f(l) >= 0) or (f(m) < 0 and f(l) < 0) ):
            l = m
        else:
            r = m

        if((r-l)/2 < t):
            return m, i
    
    return m, max_iterations

def secant_derivative(f, x0, epsilon=1e-10):
    return (f(x0+epsilon)-f(x0-epsilon))/(2*epsilon)

def newton_rhapson(f, start, t=1e-10, max_iterations = 100):

    x = start

    error = []

    for i in range(max_iterations):
        f_1 = secant_derivative(f,x)
        x_new = x - f(x)/f_1

        error.append(np.abs(x_new - x))

        x = x_new

        if(np.abs(f(x)) < t):
            return x, i, error
        
    return x, max_iterations, error
     
def line(x):
    return 2*x - 5

def random_fn(x):
    return (np.exp(-x) - x)

bisect, bitr = bisection(random_fn, -123, 123, max_iterations=50)
root, itr, error = newton_rhapson(random_fn, 2000, max_iterations=50)
print(f"Bisection: Found root at {bisect} in {bitr} iterations")
print(f"Newton Rhapson: Found root at {root} in {itr} iterations")


plt.figure()
plt.plot(np.log10(error), label = "Newton Rhapson Error")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Error vs Iteration")
plt.show()