import numpy as np
from math import pi, ceil
import scipy
from qutip import *
import c2qa
from collections import Counter
import matplotlib.pyplot as plt
from  qiskit.quantum_info import DensityMatrix

def collect_cvcircuit_metrics(circuit,cutoff):
    from collections import Counter

    # Map each qubit to its register name
    qubit_to_reg = {}
    for reg in circuit.qregs:
        for q in reg:
            qubit_to_reg[q] = reg.name

    # Separate qubit and qumode registers
    qubit_regs = []
    qumode_regs = []

    for reg in circuit.qregs:
        name = reg.name.lower()
        if any(tag in name for tag in ['qmode', 'cv', 'osc','qumode','qmr']):
            qumode_regs.append(reg)
        else:
            qubit_regs.append(reg)

    # Count qubits and qumodes by number of *registers*, not physical bits
    num_qubits = sum(len(reg) for reg in qubit_regs)
    num_qumodes = sum(len(reg) for reg in qumode_regs)  # Each reg element is one qumode
    circuit_depth = circuit.depth()

    gate_counts = Counter()
    skip_instrs = {'barrier', 'measure', 'initialize', 'snapshot', 'delay'}

    for instr, qargs, cargs in circuit.data:
        if instr.name in skip_instrs:
            continue

        involved_regs = {qubit_to_reg.get(q, "").lower() for q in qargs}
        has_qubit = any(reg.name in [r.name for r in qubit_regs] for reg in qumode_regs if reg.name.lower() in involved_regs) \
                    or any(reg in [r.name for r in qubit_regs] for reg in involved_regs)
        has_qumode = any(reg.name in [r.name for r in qumode_regs] for reg in qumode_regs if reg.name.lower() in involved_regs) \
                     or any(reg in [r.name for r in qumode_regs] for reg in involved_regs)

        if has_qubit and has_qumode:
            gate_counts['hybrid_gates'] += 1
        elif has_qubit:
            gate_counts['qubit_gates'] += 1
        elif has_qumode:
            gate_counts['qumode_gates'] += 1
        else:
            gate_counts['unknown_gates'] += 1

    return {
        "Qubits": num_qubits,
        "Qumodes": num_qumodes/int(np.ceil(np.log2(cutoff))),
        "Qubit Gates": gate_counts["qubit_gates"],
        "Qumode Gates": gate_counts["qumode_gates"],
        "Hybrid Gates": gate_counts["hybrid_gates"],
        "Circuit Depth": circuit_depth
    }

    
    
def plot_radar_metrics(metrics_list, labels=None, title="CV-DV Radar Chart"):
    keys = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates', 'Hybrid Gates', 'Total Gates']
    N = len(keys)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_vals = {key: max(metric[key] for metric in metrics_list) for key in keys}

    data = []
    for metric in metrics_list:
        normalized = [metric[key] / max_vals[key] if max_vals[key] != 0 else 0 for key in keys]
        normalized += normalized[:1]
        data.append(normalized)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)

    # Use a colormap for consistent and distinguishable colors
    colors = plt.cm.tab10.colors

    for i, d in enumerate(data):
        label = labels[i] if labels else f"Circuit {i+1}"
        color = colors[i % len(colors)]
        ax.plot(angles, d, label=label, color=color)
        ax.fill(angles, d, alpha=0.25, color=color)

        # Add text annotations for each metric value
        original_metrics = metrics_list[i]
        for j in range(N):
            angle = angles[j]
            r = d[j]
            value = original_metrics[keys[j]]
            ax.text(angle, r + 0.05, f"{value}", ha='center', va='center', fontsize=8, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
    
    
from qiskit.quantum_info import partial_trace

def get_reduced_qumode_density_matrix(stateop, qumode_index, num_qumodes, cutoff):
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    total_qubits = stateop.num_qubits

    qumode_indices = list(range(
        qumode_index * num_qubits_per_qumode,
        (qumode_index + 1) * num_qubits_per_qumode
    ))

    trace_indices = [i for i in range(total_qubits) if i not in qumode_indices]

    return partial_trace(stateop, trace_indices)


def get_reduced_qubit_density_matrix(stateop, qubit_index, num_qumodes, cutoff):
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    offset = num_qumodes * num_qubits_per_qumode
    total_qubits = stateop.num_qubits

    target_index = offset + qubit_index
    trace_indices = [i for i in range(total_qubits) if i != target_index]

    return partial_trace(stateop, trace_indices)

def wigner_negativity_all_modes(stateop, num_qumodes, cutoff, axes_min=-6, axes_max=6, axes_steps=500, g=np.sqrt(2), method="clenshaw"):
    total_negativity = 0
    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        xvec = np.linspace(axes_min, axes_max, axes_steps)
        W = c2qa.wigner._wigner(red_dm, xvec, g=g, method=method)

        dx = dy = (axes_max - axes_min) / (axes_steps - 1)
        area = np.sum(W) * dx * dy
        W /= area  # Normalize so ∫W = 1
        abs_area = np.sum(np.abs(W)) * dx * dy

        negativity = 0.5 * (abs_area - 1.0)
        negativity = min(max(negativity, 0), 1)
        total_negativity += negativity

        print(f"Mode {i}: ∫W = 1.000, ∫|W| = {abs_area:.3f}, Negativity = {negativity:.3f}")

    return total_negativity / num_qumodes

def truncation_cost_all_modes(stateop, num_qumodes, cutoff, n_tail=5):
    """
    Compute average tail probability over qumodes.
    """
    total_tail = 0
    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        diag_probs = np.real(np.diag(red_dm.data))
        tail = sum(diag_probs[-n_tail:])
        total_tail += tail

    return total_tail / num_qumodes

def average_energy_all(stateop, num_qumodes, num_qubits, cutoff, omega_qumode=1.0, omega_qubit=1.0):
    """
    Compute total energy from multiple qumodes + qubits.
    """
    E = 0

    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        n_op = num(cutoff).full()
        E += omega_qumode * np.trace(red_dm.data @ n_op).real

    for j in range(num_qubits):
        red_dm = get_reduced_qubit_density_matrix(stateop, j, num_qumodes, cutoff)
        sz = np.array([[1, 0], [0, -1]])
        E += omega_qubit * np.trace(red_dm.data @ sz).real

    return E


    
    
def evaluate_quantum_metrics(circuit, stateop, cutoff,num_qumodes=1,num_qubits=1, n_tail=5, omega_qubit=1.0, omega_qumode=1.0):
    """
    Evaluate truncation cost, Wigner total area, and average energy
    for hybrid CV-DV circuits with multiple qumodes and qubits.

    Args:
        circuit (CVCircuit): c2qa circuit
        stateop (Qiskit result object): full output state
        cutoff (int): qumode cutoff
        n_tail (int): how many highest Fock levels to include in truncation tail
        omega_qubit (float): qubit energy prefactor
        omega_qumode (float): qumode energy prefactor

    Returns:
        (truncation_cost, wigner_total_area, average_energy)
    """

    trunc = truncation_cost_all_modes(stateop, num_qumodes, cutoff, n_tail=n_tail)
    wigner_area = wigner_negativity_all_modes(stateop, num_qumodes, cutoff)
    avg_energy = average_energy_all(stateop, num_qumodes, num_qubits, cutoff, omega_qubit=omega_qubit, omega_qumode=omega_qumode)

    return trunc, wigner_area, avg_energy