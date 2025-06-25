import numpy as np
from tqdm import tqdm
from simulation import run_single_trial
from analysis import plot_phase_transition

def main():
    # --- Simulation Parameters ---
    N = 100                  # Dimensionality (sufficient to see sharp transition)
    KAPPA = 1.0              # Margin parameter (you can experiment with this)  
    NUM_TRIALS = 20          # Number of trials for each alpha to average over
    ALPHA_MIN = 0.2
    ALPHA_MAX = 0.8          # Test a range around the critical point
    ALPHA_STEPS = 20

    # --- Theoretical Prediction ---
    alpha_c_theoretical = 1.0 / (KAPPA**2 + 1.0)
    print(f"Parameters: N={N}, kappa={KAPPA}, num_trials={NUM_TRIALS}")
    print(f"Theoretical critical capacity alpha_c = {alpha_c_theoretical:.4f}")

    # --- Run Simulation ---
    alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
    success_rates = []

    for alpha in tqdm(alpha_values, desc="Simulating"):
        P = int(alpha * N)
        success_count = 0
        for _ in range(NUM_TRIALS):
            if run_single_trial(N, P, KAPPA):
                success_count += 1
        success_rates.append(success_count / NUM_TRIALS)
    
    # --- Plotting ---
    plot_phase_transition(alpha_values, success_rates, alpha_c_theoretical, N, KAPPA)

if __name__ == "__main__":
    main()