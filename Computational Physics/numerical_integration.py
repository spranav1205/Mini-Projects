import numpy as np

def trapezoidal(function, left, right, step_size=1e-5):

    sum = 0

    sum += function(left)
    sum += function(right)

    max_step = np.abs(np.int64((np.float64(right-left)/np.float64(step_size))))
    print(max_step)

    flag = 0

    if (right<left):
        temp = left
        left = right
        right = temp
        flag = 1

    for i in range(1,max_step):
        
        x = left + i*step_size
        y = function(x)

        sum += y*2  

    if (flag == 0):
        return sum*step_size/2
    else:
        return -sum*step_size/2
    
def simpson(function, left, right, step_size=1e-5):
    sum = 0.0

    sum += function(left)
    sum +=function(right)

    max_step = np.abs(np.int64((np.float64(right-left)/np.float64(step_size))))
    print(max_step)

    flag = 0

    if(right<left):
        temp = left
        left = right
        right = temp
        flag = 1

    for i in range(1,max_step):
        if(i%2):
            x = left + i*step_size
            y = function(x)
            sum += y*4 
        
        else:
            x = left + i*step_size
            y = function(x)
            sum += y*2

    if (flag == 0):
        return sum*step_size/3
    else:
        return -sum*step_size/3
            
    
def random_fn(x):
    # return np.exp(-x)

    return x**3 - 4*x**2 + 12*x - 23

def square(x):
    return x*x

print(trapezoidal(random_fn, -1, 1))
print(simpson(random_fn, 1, -1))

