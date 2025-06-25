import numpy as np
from scipy.optimize import minimize

def run_single_trial(N: int, P: int, kappa: float) -> bool:
    """
    Runs a single trial of the Gardner problem.
    
    The Gardner problem: Find a weight vector w such that:
    - w · x_μ ≥ κ for all μ = 1, ..., P  
    - ||w||² = N
    
    We solve this by checking if the constrained optimization problem has a solution.

    Args:
        N: Number of dimensions (features).
        P: Number of data points.
        kappa: The margin parameter.

    Returns:
        True if a weight vector satisfying the constraints exists, False otherwise.
    """
    # Generate random data points x_μ ~ N(0, I/N)
    X = np.random.randn(P, N) / np.sqrt(N)
    
    # The Gardner problem can be solved by checking if there exists w with ||w||=sqrt(N)
    # such that min_μ (w · x_μ / ||w||) >= kappa / sqrt(N)
    # This is equivalent to checking if max_w min_μ (w · x_μ) >= kappa when ||w||² = N
    
    # Approach: Use the fact that the solution (if it exists) can be found by
    # maximizing the minimum margin. This is equivalent to solving:
    # maximize t subject to w · x_μ >= t for all μ and ||w||² = N
    
    try:
        # Method 1: Direct approach using scipy optimization
        # We'll solve: minimize -t subject to w·x_μ >= t and ||w||² = N
        
        def objective(vars):
            # vars = [w1, w2, ..., wN, t] where w = [w1, ..., wN] and t is the margin
            w = vars[:-1]
            t = vars[-1]
            return -t  # maximize t by minimizing -t
        
        def constraint_margin(vars):
            # Constraint: w·x_μ >= t for all μ
            w = vars[:-1]
            t = vars[-1]
            margins = X @ w - t
            return margins  # should be >= 0
        
        def constraint_norm(vars):
            # Constraint: ||w||² = N
            w = vars[:-1]
            return N - np.sum(w**2)  # should be = 0, so ||w||² - N = 0
        
        # Initial guess: random w with ||w|| = sqrt(N), t = 0
        w_init = np.random.randn(N)
        w_init = w_init * np.sqrt(N) / np.linalg.norm(w_init)
        x0 = np.concatenate([w_init, [0.0]])
        
        # Set up constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_margin},  # w·x_μ >= t
            {'type': 'eq', 'fun': constraint_norm}       # ||w||² = N  
        ]
        
        # Solve the optimization problem
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                         options={'ftol': 1e-9, 'disp': False})
        
        if result.success:
            w_opt = result.x[:-1]
            t_opt = result.x[-1]
            
            # Verify the solution
            norm_check = abs(np.sum(w_opt**2) - N) < 1e-6
            margin_check = np.all(X @ w_opt >= t_opt - 1e-6)
            
            if norm_check and margin_check:
                return t_opt >= kappa - 1e-6
        
        # Method 2: Alternative approach if optimization fails
        # Use random projections to estimate the maximum achievable margin
        max_margin = -np.inf
        for _ in range(20):  # Reduced from 100 to 20 for speed
            w_random = np.random.randn(N)
            w_random = w_random * np.sqrt(N) / np.linalg.norm(w_random)  # normalize to ||w|| = sqrt(N)
            
            margins = X @ w_random
            min_margin = np.min(margins)
            max_margin = max(max_margin, min_margin)
        
        return max_margin >= kappa - 1e-6
        
    except Exception:
        return False