import numpy as np

def gaussian_elimination(A, b):
    
    if(len(A.shape) != 2 or A.shape[0] != b.shape[0]):
        raise Exception
    
    n = A.shape[0]

    for i in range(n-1):
        for j in range(i+1,n):
            
            if(A[j][i] == 0):
                break
            
            else:
                factor = A[j][i]/A[i][i]
                A[j] = A[j] - A[i]*factor
                b[j] = b[j] - b[i]*factor 

    return A,b

def back_substitution(A,b):
    x = np.empty_like(b)
    n = b.shape[0]

    for i in range(n-1,-1, -1):
        factor = 1/A[i][i]

        rhs = b[i]

        if (i != n-1):
            for j in range(i+1, n):
                rhs -= A[i][j]*x[j]

        x[i] = rhs*factor

    return x



if __name__ == "__main__":
    b = np.array([10,5,15,6], dtype=float)
    A = np.array([[2,1,-1,3],[1,3,2,-1],[3,2,4,1],[1,2,0,2]],dtype=float)
    print(gaussian_elimination(A,b))

    A,b = gaussian_elimination(A,b)
    print(back_substitution(A,b))