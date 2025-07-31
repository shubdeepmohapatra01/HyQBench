import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fractions import Fraction
from math import gcd
from qutip import *

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PARENT_DIR)

from custom_gates import shors, state_generation
from benchmarks_circuit import gkp_state_circuit,shors_circuit
import c2qa

# === SETTINGS ===
RUN_ON_SERVER = False  # Set to False for local debug
RESULTS_DIR = "results_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = os.path.join(RESULTS_DIR, f"log_{timestamp}.txt")
factors_file = os.path.join(RESULTS_DIR, f"factors_{timestamp}.txt")

def write_log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")

def write_factors(msg):
    with open(factors_file, "a") as f:
        f.write(msg + "\n")

# === UTILITIES ===

def try_find_factors(N, r, a):
    if r % 2 != 0:
        return None
    candidate = pow(a, r//2, N)
    if candidate == N-1 or candidate == 1:
        return None
    p = np.gcd(candidate-1, N)
    q = np.gcd(candidate+1, N)
    if p*q == N and p != 1 and q != 1:
        return (p, q)
    return None

def find_valid_a_values(N):
    valid_a = []
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        r = 1
        while pow(a, r, N) != 1 and r < N:
            r += 1
        if r % 2 == 0 and pow(a, r // 2, N) != N - 1:
            valid_a.append((a, r))
    return valid_a

def sample_p_and_estimate_period(p_dist, paxis, max_denominator=100):
    p_dist = np.nan_to_num(p_dist, nan=0.0, posinf=0.0, neginf=0.0)
    p_dist = np.clip(p_dist, 0, None)  

    total = np.sum(p_dist)
    if total == 0:
        raise ValueError("Probability distribution sums to zero after sanitization.")

    prob_dist = p_dist / total

    p_sample = np.random.choice(paxis, p=prob_dist)

    # Estimate period
    s_over_r = p_sample / (2 * np.pi)
    frac = Fraction(s_over_r).limit_denominator(max_denominator)
    j, r = frac.numerator, frac.denominator

    estimated_period = r if gcd(j, r) == 1 else None

    return estimated_period, (j, r), p_sample

def generate_gkp_codeword(cutoff, delta=0.3, kappa=1.0, logical=0, num_peaks=7):
    a = destroy(cutoff)
    sq = squeeze(cutoff, -np.log(delta))

    state = 0
    spacing = np.sqrt(np.pi)
    for k in range(-num_peaks//2, num_peaks//2 + 1):
        shift = (2 * k + logical) * spacing
        envelope = np.exp(-0.5 * (k * kappa)**2)
        disp = displace(cutoff, shift)
        peak = disp * sq * basis(cutoff, 0)
        state += envelope * peak

    state = state.unit()
    return state

# === MAIN FUNCTION ===

def estimate_success_probability(N, m, R, delta, cutoff, trials=30, shots=1024):
    qmr1 = c2qa.QumodeRegister(num_qumodes=3, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1)
    cr = ClassicalRegister(1)

    valid_a_r_pairs = find_valid_a_values(N)
    if not valid_a_r_pairs:
        write_log(f"[N={N}] No valid a values found.")
        return 0.0, [], 0, 0

    total_successes = 0
    total_shots = 0
    all_factors = set()
    alpha = np.sqrt(np.pi)

    for trial in range(trials):
        a, true_r = random.choice(valid_a_r_pairs)
        write_log(f"[Trial {trial+1}] a = {a}, expected r = {true_r}")

        circuit = c2qa.CVCircuit(qmr1, qbr, cr)
        
        circuit = shors_circuit(N,m,R,a,delta,cutoff)


        # Run once to get momentum distribution
        stateop, _, _ = c2qa.util.simulate(circuit, shots=1)
        rho_qumode_0 = shors.get_reduced_qumode_density_matrix(stateop, qumode_index=0, num_qumodes=3, cutoff=cutoff)
        x_dist, xaxis = shors.momentum_plotting(rho_qumode_0, cutoff, ax_min=-30, ax_max=30, steps=200)

        for _ in range(shots):
            estimated_r, (j, r), p_sample = sample_p_and_estimate_period(x_dist.flatten(), xaxis)

            if estimated_r is not None:
                factors = try_find_factors(N, estimated_r, a)
                if factors:
                    total_successes += 1
                    all_factors.update(factors)
                    write_log(f"Shot success: r={estimated_r}, factors={factors}")
                    write_factors(f"N={N}, a={a}, r={estimated_r}, factors={factors}")
                else:
                    write_log(f"Shot fail: r={estimated_r} gave no valid factors.")
            else:
                write_log("Shot fail: Could not extract valid r.")
            total_shots += 1

    success_rate = total_successes / total_shots if total_shots > 0 else 0.0
    write_log(f"[N={N}, cutoff={cutoff}] Total success probability: {success_rate:.4f}")
    return success_rate, sorted(all_factors), total_successes, total_shots



