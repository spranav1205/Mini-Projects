import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ODE import lorenz_attractor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
sigma, beta, rho = 10, 8/3, 28
X = lambda x,y,z: sigma * (y - x)
Y = lambda x,y,z: x * (rho - z) - y
Z = lambda x,y,z: x * y - beta * z

# Main trajectory
x0, y0, z0 = 0,0,0
time, step = 200, 0.01
x, y, z, t = lorenz_attractor(X, Y, Z, x0, y0, z0, time, step)

# Ensemble initial conditions (spread out)
n_particles = 50
spread = 5
np.random.seed(1)
init_conditions = np.array([ [x0, y0, z0] + spread*np.random.randn(3) 
                              for _ in range(n_particles) ])

trajectories = [lorenz_attractor(X, Y, Z, xi, yi, zi, time, step)
                for (xi, yi, zi) in init_conditions]

# Convert for easier indexing
steps = len(x)
xs = np.array([traj[0] for traj in trajectories]).T
ys = np.array([traj[1] for traj in trajectories]).T
zs = np.array([traj[2] for traj in trajectories]).T

# --- Animation ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-25, 25])
ax.set_ylim([-35, 35])
ax.set_zlim([0, 50])
ax.set_title("Lorenz Attractor with Ensemble")

# Main butterfly trajectory (blue)
line_main, = ax.plot([], [], [], lw=1.0, color="blue")

# Ensemble trails (red)
lines = [ax.plot([], [], [], lw=0.7, color="red", alpha=0.6)[0] for _ in range(n_particles)]

# Limit tail length for speed
tail = 500  

def init():
    line_main.set_data([], [])
    line_main.set_3d_properties([])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return [line_main] + lines

def update(frame):
    # Main butterfly (full trail)
    line_main.set_data(x[:frame], y[:frame])
    line_main.set_3d_properties(z[:frame])

    # Ensemble trails (short tails)
    start = max(0, frame-tail)
    for i, line in enumerate(lines):
        line.set_data(xs[start:frame, i], ys[start:frame, i])
        line.set_3d_properties(zs[start:frame, i])
    return [line_main] + lines

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    interval=1, blit=True)
plt.show()
