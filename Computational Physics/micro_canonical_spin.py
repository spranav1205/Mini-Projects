# --------------------------------------------------------------
# Compare microcanonical vs canonical entropy per spin
# for a system of N two-level spins with energies ±alpha
#
# Goal: show that as N increases, microcanonical entropy per spin
# converges to the canonical entropy per spin
#
# --------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt
from math import lgamma

# --------------------------------------------------------------
# Utility function: compute log of "n choose k"
# using the gamma function for numerical stability
# --------------------------------------------------------------
def log_binomial(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

# --------------------------------------------------------------
# Microcanonical entropy per spin
# n : number of spins
# b : shorthand for beta*alpha = (1/kT)*alpha
#
# We pick the value of k (number of up spins) that corresponds
# to the canonical mean magnetization. Then compute the binomial
# degeneracy at that k, and take log/binomial/n.
# --------------------------------------------------------------
def micro_entropy_per_spin(n, b):
    # canonical average magnetization per spin
    m = -math.tanh(b)

    # expected number of up spins corresponding to that m
    k = int(round(n * (1 + m) / 2.0))

    # keep k within [0, n]
    k = max(0, min(n, k))

    return log_binomial(n, k) / n

# --------------------------------------------------------------
# Canonical entropy per spin
# Derived directly from the partition function
# --------------------------------------------------------------
def canonical_entropy_per_spin(b):
    return math.log(2 * math.cosh(b)) - b * math.tanh(b)

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
Ns = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
bs = [0.2, 1.0, 2.0]   # different values of beta*alpha

# Collect results
results = []

for b in bs:
    S_can = canonical_entropy_per_spin(b)
    for n in Ns:
        S_micro = micro_entropy_per_spin(n, b)
        results.append((b, n, S_micro, S_can, S_micro - S_can))

# --------------------------------------------------------------
# Plot convergence
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
for b in bs:
    # filter rows for this b
    rows = [r for r in results if r[0] == b]
    n_values = [r[1] for r in rows]
    S_micro_values = [r[2] for r in rows]
    S_can_value = rows[0][3]  # same for all n at fixed b

    plt.plot(n_values, S_micro_values, marker='o',
             label=f"micro per spin (b={b})")
    plt.hlines(S_can_value, xmin=min(Ns), xmax=max(Ns),
               linestyles='dashed', label=f"canonical per spin (b={b})")

plt.xscale('log')
plt.xlabel('Number of spins (log scale)')
plt.ylabel('Entropy per spin (units of k_B)')
plt.title('Convergence of microcanonical entropy to canonical entropy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
