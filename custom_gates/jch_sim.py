import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate

def sigmax_(beta,j,n,cutoff):
    qubit_pauli = tensor([identity(2)]*(j-1) + [sigmax()] + [identity(2)]*(n-j))
    # print(qubit_pauli)
    qumode_matrix = tensor([identity(cutoff)]*(j-1) + [1j * ((beta/2)*(destroy(cutoff) + create(cutoff)))] + [identity(cutoff)]*(n-j))
    return tensor(qubit_pauli, qumode_matrix).expm()

def sigmay_(beta,j,n,cutoff):
    qubit_pauli = tensor([identity(2)]*(j-1) + [sigmay()] + [identity(2)]*(n-j))
    # print(qubit_pauli)
    qumode_matrix = tensor([identity(cutoff)]*(j-1) + [ (1j * -1j *(beta/2)*(create(cutoff) - destroy(cutoff)))] + [identity(cutoff)]*(n-j))
    return tensor(qubit_pauli, qumode_matrix).expm()

def coupling_term(beta,n,j,cutoff):
    U = sigmax_(beta,j+1,n,cutoff) * sigmay_(beta,j+1,n,cutoff) 
    return U

def createCircuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,display_circuit = False):
    # Initialize registers
    qmr = c2qa.QumodeRegister(num_qumodes=Nsites, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))))
    qbr = QuantumRegister(Nqubits)
    circuit = c2qa.CVCircuit(qmr, qbr)

    # Hopping interaction between adjacent qumodes
    theta = -J * tau
    for i in range(0, len(qmr) - 1, 2):
        circuit.cv_bs(theta, qmr[i], qmr[i + 1])
    for i in range(1, len(qmr) - 1, 2):
        circuit.cv_bs(theta, qmr[i], qmr[i + 1])

    # Local resonator evolution
    theta_resonator = omega_r * tau
    for i in range(len(qmr)):
        circuit.cv_r(theta_resonator, qmr[i])

    # Local qubit evolution
    theta_qubit = omega_q * tau
    for i in range(len(qbr)):
        circuit.rz(theta_qubit, qbr[i])

    # Qubit-qumode coupling
    theta_coupling = g * tau
    for i in range(len(qbr)):
        circuit.cv_jc(theta_coupling,0,qmr[i],qbr[i])
        # coupling = coupling_term(theta_coupling, Nqubits, i,cutoff)
        # coupling_gate = UnitaryGate(coupling.full(), label=f'Coupling_{i}')
        # circuit.append(coupling_gate, qmr[:] + qbr[:])

    # Draw the circuit for visualization
    if(display_circuit):
        circuit.draw('mpl')
    
    # Convert to a gate for modular use
    return circuit_to_gate(circuit, label='U')

def circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,timesteps,display_circuit = False):
    qmr = c2qa.QumodeRegister(num_qumodes=Nsites, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),name = 'qumode')
    qbr = QuantumRegister(Nqubits)
    circuit = c2qa.CVCircuit(qmr, qbr)
    
    for i in range(timesteps):
        # Hopping interaction between adjacent qumodes
        theta = -J * tau
        for i in range(0, len(qmr) - 1, 2):
            circuit.cv_bs(theta, qmr[i], qmr[i + 1])
        for i in range(1, len(qmr) - 1, 2):
            circuit.cv_bs(theta, qmr[i], qmr[i + 1])

        # Local resonator evolution
        theta_resonator = omega_r * tau
        for i in range(len(qmr)):
            circuit.cv_r(theta_resonator, qmr[i])

        # Local qubit evolution
        theta_qubit = omega_q * tau
        for i in range(len(qbr)):
            circuit.rz(theta_qubit, qbr[i])

        # Qubit-qumode coupling
        theta_coupling = g * tau
        for i in range(len(qbr)):
            circuit.cv_jc(theta_coupling,0,qmr[i],qbr[i])
    
    return circuit