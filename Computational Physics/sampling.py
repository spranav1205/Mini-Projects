import matplotlib.pyplot as plt

class generator:
    def __init__(self,a,b, dtype=float):
        self.lower = a
        self.upper = b

        self.a = 1103515245
        self.c = 12345
        self.big = 2**31 - 1

    def lcg(self, n, seed = 123):

        count = 0
        numbers = [seed]
        x = seed

        for i in range(n):
            x = (self.a*x + self.c)%self.big
            numbers.append(x)

        for i in range(len(numbers)):
            numbers[i] = self.lower + (self.upper - self.lower) * numbers[i] / self.big

            if (type(numbers[i]) == int):
                numbers[i] = int(numbers[i])

        return numbers
    
def autorcorelation(data, lag):

    n = len(data)
    mean = sum(data) / n
    c0 = sum((x - mean) ** 2 for x in data) / n

    def r(k):
        return sum((data[i] - mean) * (data[i + k] - mean) for i in range(n - k)) / n / c0

    return [r(k) for k in range(lag)]

    
if __name__ == "__main__":
    
    gen = generator(0, 100)
    numbers = gen.lcg(100000, seed=42)

    print(numbers[:20])

    plt.hist(numbers, bins=100)
    plt.title("LCG Generated Numbers")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    ac = autorcorelation(numbers, 100)
    plt.plot(ac)
    plt.title("Autocorrelation of LCG Generated Numbers")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()