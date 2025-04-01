import numpy as np

def gauss_seidel(A, b, x0, iterations):
    n = len(b)
    x = np.copy(x0)
    
    print("Initial guess:", x0)
    print("\nPerforming Gauss-Seidel iterations:\n")
    
    for k in range(iterations):
        print(f"Iteration {k+1}:")
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))  # Use updated values
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # Use old values
            old_xi = x[i]
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
            print(f"  x[{i+1}] = (b[{i+1}] - ({sum1:.6f}) - ({sum2:.6f})) / A[{i+1},{i+1}] = {x[i]:.6f} (previously {old_xi:.6f})")
        print("  Updated solution:", x, "\n")
    
    return x

# Define the system Ax = b
A = np.array([[1, -1],
              [-1, 2]], dtype=float)
b = np.array([1, 1], dtype=float)

# Initial guess
x0 = np.array([0, 0], dtype=float)

# Perform 2 iterations
gauss_seidel(A, b, x0, iterations=2)
