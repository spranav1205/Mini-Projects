import numpy as np
import matplotlib.pyplot as plt

def linear_ODE(f,y0, t0, iterations = 1000, step = 1e-3):
    """
    dy / dx = f
    y0 = Initial Conditions
    """
    t_points = []
    y_points = []

    y = y0
    t = t0

    for i in range(iterations):
        
        y_points.append(y) 
        t_points.append(t)

        y = y + step * f(y,t)
        t = t + step

    return y_points, t_points

def rk4(f, y0, t0, iterations = 1000, step = 1e-3):
    """
    dy / dx = f
    y0 = Initial Conditions
    """
    t_points = []
    y_points = []

    y = y0
    t = t0

    for i in range(iterations):
        
        y_points.append(y) 
        t_points.append(t)

        k1 = f(y ,t)
        k2 = f(y+k1*step/2, t+step/2)
        k3 = f(y+k2*step/2, t+step/2)
        k4 = f(y+k3*step, t+step)

        t = t + step
        y = y + step/6 * (k1 + 2*k2 + 2*k3 + k4)

    return y_points, t_points

def rk4_2(f, y1_0, y2_0, t0, iterations = 1000, step = 1e-3):
    """
    y1' = y2
    y1'' = y2' = f
    
    let y1 be a spatial co-ordinate
    y2 is thus the velocity equivalent
    f is the acceleration

    h is our time step

    """
    
    t_points = []
    y1_points = []
    y2_points = []

    y2 = y2_0
    y1 = y1_0
    t = t0

    h = step

    for i in range(iterations):
        
        y2_points.append(y2)
        y1_points.append(y1) 
        t_points.append(t)

        k1 = h * y2           # Change is position
        l1 = h * f(t, y1, y2) # Change in Velocity

        k2 = h * (y2 + l1/2)                # Updated Position
        l2 = h * f(t+h/2, y1+k1/2, y2+l1/2) # Updated Velocity and Position used to calculate acceleration 

        k3 = h * (y2 + l2/2)
        l3 = h * f(t+h/2, y1+k2/2, y2+l2/2)

        k4 = h * (y2 + l3)
        l4 = h * f(t+h, y1+k3, y2+l3)

        y1 = y1 + (1/6)*(k1+2*k2+2*k3+k4)
        y2 = y2 + (1/6)*(l1+2*l2+2*l3+l4)

        t = t + h
    
    return y1_points, y2_points, t_points

def damped_oscillator(t, y1, y2, F = 10, w = 2, gamma = 0.1):
    
    """
    y2 is y1'
    """

    return (F * np.cos(w*t) - gamma * y2 - y1)

def lorenz_attractor(X, Y, Z, x0, y0, z0, time=10, step = 10e-3):

    iter = int(time / step)
    
    t_points = []
    x_points = []
    y_points = []
    z_points = []

    t = t0
    x = x0
    y = y0
    z = z0

    h = step

    for i in range(iter):

        x_points.append(x)
        y_points.append(y)
        z_points.append(z)
        t_points.append(t)

        k1 = X(x,y,z) * step
        l1 = Y(x,y,z) * step
        m1 = Z(x,y,z) * step

        k2 = X(x+k1/2,y+l1/2,z+m1/2) * step
        l2 = Y(x+k1/2,y+l1/2,z+m1/2) * step
        m2 = Z(x+k1/2,y+l1/2,z+m1/2) * step

        k3 = X(x+k2/2,y+l2/2,z+m2/2) * step
        l3 = Y(x+k2/2,y+l2/2,z+m2/2) * step
        m3 = Z(x+k2/2,y+l2/2,z+m2/2) * step

        k4 = X(x+k3,y+l3,z+m3) * step
        l4 = Y(x+k3,y+l3,z+m3) * step
        m4 = Z(x+k3,y+l3,z+m3) * step

        x = x + h * (k1 + 2*k2 + 2*k3 + k4)
        y = y + h * (l1 + 2*l2 + 2*l3 + l4)
        z = z + h * (m1 + 2*m2 + 2*m3 + m4)
        t = t + h
    
    return x_points, y_points, z_points, t_points



if __name__ == "__main__":
    
    # Simple test of RK4
    # F = 1
    # w = 1.1
    # gamma = 0

    # y1_0 = 10
    # y2_0 = 2
    # t0 = 0

    # iterations = 100000
    # step = 1e-3

    # y1_points, y2_points, t_points = rk4_2(lambda t, y1, y2: damped_oscillator(t, y1, y2, F, w, gamma), y1_0, y2_0, t0, iterations, step)

    # plt.subplot(3,1,1)
    # plt.plot(t_points, y1_points)
    # plt.title("Position vs Time")
    # plt.xlabel("Time")
    # plt.ylabel("Position")  
    # plt.grid()
    # plt.subplot(3,1,2)
    # plt.plot(t_points, y2_points)
    # plt.title("Velocity vs Time")
    # plt.xlabel("Time")
    # plt.ylabel("Velocity")  
    # plt.grid()
    # plt.subplot(3,1,3)
    # axes = plt.gca()
    # axes.set_aspect("auto")
    # plt.plot(y1_points, y2_points)
    # plt.title("Phase Space")
    # plt.xlabel("Position")
    # plt.ylabel("Velocity")  
    # plt.grid()  
    # plt.tight_layout()
    # plt.show()

    # Lorenz Attractor Test
    sigma = 10  
    beta = 8/3
    rho = 28

    X = lambda x,y,z: sigma * (y - x)
    Y = lambda x,y,z: x * (rho - z) - y
    Z = lambda x,y,z: x * y - beta * z

    x0, y0, z0 = 21, 100, -15
    time = 500
    step = 1e-2
    t0 = 0
    
    x_points, y_points, z_points, t_points = lorenz_attractor(X, Y, Z, x0, y0, z0, time, step)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_points, y_points, z_points, lw=0.5)
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()