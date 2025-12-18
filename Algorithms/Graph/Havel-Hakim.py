import numpy as np

def havel_hakim(degrees: list):
    
    while (len(degrees)>0):
        degrees.sort(reverse=True)

        if(degrees[-1] < 0):
            return False

        d = degrees.pop(0)
        for i in range(d):
            degrees[i] -= 1

    return True

if __name__ == "__main__":
    deg_seq = [4, 3, 3, 2, 2]
    print(havel_hakim(deg_seq))