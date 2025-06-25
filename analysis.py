import matplotlib.pyplot as plt
import numpy as np

def plot_phase_transition(alpha_values: np.ndarray, 
                          success_rates: list, 
                          alpha_c_theoretical: float,
                          N: int,
                          kappa: float):
    """
    Plots the success rate vs. alpha and the theoretical prediction.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot simulation results
    ax.plot(alpha_values, success_rates, 'o-', label='Numerical Simulation Results', color='b', markersize=5)

    # Plot theoretical prediction
    ax.axvline(x=alpha_c_theoretical, color='r', linestyle='--', 
               label=f'Theoretical α_c = {alpha_c_theoretical:.3f}')

    # Formatting
    ax.set_xlabel('Capacity α = P/N', fontsize=14)
    ax.set_ylabel('Success Rate (Probability of Finding a Solution)', fontsize=14)
    ax.set_title(f'Gardner Capacity Phase Transition (N={N}, κ={kappa})', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(left=min(alpha_values), right=max(alpha_values))
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'gardner_capacity_N{N}_kappa{kappa}.png', dpi=150, bbox_inches='tight')
    plt.show()