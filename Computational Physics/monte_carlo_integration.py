import random
import matplotlib.pyplot as plt

class monte_carlo_integrator:
    def __init__(self, f):
        self.function = f

    def integrate(self, samples: int, a, b): 

        sum = 0.0
        
        for _ in range(samples):
            f = random.uniform(a, b)
            sum += self.function(f)

        return sum * (b-a)/samples
    
    def get_interval(self, samples: int, a, b, iteration = 100, plot = False):

        vals = []

        for i in range(iteration):
            vals.append(self.integrate(samples, a, b))

        vals.sort()

        min = iteration * 0.005
        max = iteration * 0.995

        if plot:
            plt.hist(vals, bins=20)
            plt.title("Monte Carlo Integration Results")
            plt.xlabel("Integral Value")
            plt.ylabel("Frequency")
            plt.show()

        mean = sum(vals) / iteration
        variance = sum((x - mean) ** 2 for x in vals) / iteration
        std = variance ** 0.5

        return vals[int(min)], vals[int(max)], std

class monte_carlo_multivariate_integrator:
    def __init__(self, f, dimensions: int):
        self.function = f
        self.dimensions = dimensions

    def integrate(self, samples: int, bounds: list): 

        sum = 0.0
        volume = 1.0
        
        for i in range(self.dimensions):
            volume *= (bounds[i][1] - bounds[i][0])

        for _ in range(samples):
            point = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(self.dimensions)]
            sum += self.function(point)

        return sum * volume / samples

    
if __name__ == "__main__":
    def my_function(x):
        return 4/(1 + x**2)

    # integrator = monte_carlo_integrator(my_function)
    # result = integrator.integrate(100000, 0, 1)
    # print(f"Estimated integral: {result}")

    # errors = []

    # for samples in [100, 1000, 10000, 100000]:
    #     lower, upper, std = integrator.get_interval(samples, 0, 1, iteration=2000)

    #     max_error = 3.14159265358979359 - lower
    #     min_error = upper - 3.14159265358979359

    #     max_absolute_error = max(abs(max_error), abs(min_error))
    #     errors.append((samples, max_absolute_error))

    #     print(f"Samples: {samples}, 99% Interval: [{lower}, {upper}], Std Dev: {std}, Max Absolute Error: {max_absolute_error}")
    
    # plt.figure()
    # sample_sizes = [e[0] for e in errors]
    # error_values = [e[1] for e in errors]

    # plt.loglog(sample_sizes, error_values, marker='o')
    # plt.title("Error vs Sample Size in Monte Carlo Integration")
    # plt.xlabel("Sample Size")
    # plt.ylabel("Max Absolute Error")
    # plt.grid()
    # plt.show()

    def my_multivariate_function(point):
        if (point[0]**2 + point[1]**2 + point[2]**2) <= 1:
            return 1
        return 0

    multivariate_integrator = monte_carlo_multivariate_integrator(my_multivariate_function, 3)
    result = multivariate_integrator.integrate(100000000, [[-1, 1], [-1, 1], [-1, 1]])
    print(f"Estimated volume of unit sphere: {result}")

    # The exact volume of a unit sphere in 3D is (4/3) * pi
    exact_volume = (4/3) * 3.14159265358979359
    print(f"Exact volume of unit sphere: {exact_volume}")