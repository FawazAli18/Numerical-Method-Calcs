import numpy as np

def conjugate_gradient(A, b , x0, max_iter = 100, tol= 1e-6):

    x = x0.copy()
    r = b - A @ x0
    p = r.copy()
    residuals = [float(np.linalg.norm(r))]

    for k in range (max_iter):
        Ap= A @ p
        alpha = (r.T @ r) / (p.T @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
   

        #Convergence check
        current_residuals = float(np.linalg.norm(r_new))
        residuals.append(current_residuals)
        if current_residuals < tol:
            return x, True, k+1, residuals
        
        beta = float((r_new.T @ r_new) / (r.T @ r))
        p = r_new + beta * p
        r = r_new

    return x, False, max_iter, residuals


#GIven System
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=float)
b= np.array([[1, 2, 3]], dtype = float).reshape(-1, 1)
x0 = np.array([[3, 3, 4]], dtype=float).reshape(-1,1)

#solving with cg
solution, converged, num_iter, residuals = conjugate_gradient(A, b, x0)

print("Solution found:", solution)
print("Converged:", converged)
print("Number of iterations:", num_iter)
print("Final residual:", residuals[-1])

# Verify by direct solution
exact_solution = np.linalg.solve(A, b)
print("\nExact solution (for verification):", exact_solution)
print("Error norm:", np.linalg.norm(solution - exact_solution))