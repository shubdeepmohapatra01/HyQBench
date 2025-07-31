import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PARENT_DIR)
import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from custom_gates import state_generation,state_transfer,jch_sim,bosonic_vqe_new,shors_bq,bosonic_qaoa
from scipy.optimize import minimize

def cat_state_circuit(cutoff,circuit,qbr,qmr,alpha):
    circuit.h(qbr[0])
    circuit.cv_c_d(alpha / np.sqrt(2),qmr[0],qbr[0])
    circuit.h(qbr[0])

    circuit.sdg(qbr[0])
    circuit.h(qbr[0])
    circuit.cv_c_d(1j*np.pi/(8*alpha*np.sqrt(2)),qmr[0],qbr[0])
    circuit.h(qbr[0])
    circuit.s(qbr[0])
    
    return circuit

def gkp_state_circuit(cutoff,circuit,qbr,qmr,N_rounds=9,r=0.222,i=0):
    alpha = np.sqrt(np.pi)
    circuit.cv_sq(r,qmr[i])
    for k in range(1,N_rounds):
        circuit.h(qbr[0])
        circuit.cv_c_d(alpha / np.sqrt(2),qmr[i],qbr[0])
        circuit.h(qbr[0])

        circuit.sdg(qbr[0])
        circuit.h(qbr[0])
        circuit.cv_c_d(1j*np.pi/(8*alpha*np.sqrt(2)),qmr[i],qbr[0])
        circuit.h(qbr[0])
        circuit.s(qbr[0])
        
    return circuit

def apply_basis_transformation(circuit, qbr1):
    num_qubits = len(qbr1)
    for i in range(num_qubits):
        circuit.h(qbr1[i])
        if i == num_qubits - 1:  # MSB
            circuit.x(qbr1[i])
            circuit.z(qbr1[i])
        elif i == 0:  # LSB
            circuit.z(qbr1[i])
        else:  # Middle qubits
            circuit.x(qbr1[i])
            
def apply_basis_transformation_reverse(circuit, qbr1):
    num_qubits = len(qbr1)
    for i in range(num_qubits):
        if i == num_qubits - 1:  # MSB
            circuit.z(qbr1[i])
            circuit.x(qbr1[i])
            circuit.h(qbr1[i])
        elif i == 0:  # LSB
            circuit.z(qbr1[i])
            circuit.h(qbr1[i])
        else:  # Middle qubits
            circuit.x(qbr1[i])
            circuit.h(qbr1[i])

def state_transfer_CVtoDV(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(1,n+1):
        V_j = state_transfer.Vj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        
        W_j = state_transfer.Wj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction
        
    apply_basis_transformation(circuit, qbr)

    # Simulate and measure
    for i in range(n):
        circuit.measure(qbr[i], cr[-(i + 1)])
        
    return circuit

def state_transfer_DVtoCV(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(n+1,0,-1):
        W_j = state_transfer.Wj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction 
        
        V_j = state_transfer.Vj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        
    # Simulate and measure
    for i in range(n):
        circuit.measure(qbr[i], cr[-(i + 1)])
        
    return circuit

def JCH_simulation_circuit_unitary(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau):
    U1 = jch_sim.createCircuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau)
    
    return U1

def JCH_simulation_circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,timesteps):
    circuit = jch_sim.circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau,timesteps)
    
    return circuit

def binary_knapsack_vqe(H, ndepth, nfocks, maxiter=100, method='COBYLA', verb=0,threshold=1e-08, print_freq=10, Xvec=[]):
    en, Xvec, int_results = bosonic_vqe_new.ecd_opt_vqe(H, ndepth, nfocks, maxiter=maxiter, method='BFGS',
                                    verb=1, threshold=1e-9)
    
    return en,Xvec,int_results

def binary_knapsack_vqe_circuit(H, ndepth, nfocks,Xvec=[]):
    # Bound parameters
    beta_mag_min = 0.0
    beta_mag_max = 10.0
    beta_arg_min = 0.0
    beta_arg_max = 2 * np.pi
    theta_min = 0.0
    theta_max = np.pi
    phi_min = 0.0
    phi_max = 2 * np.pi

    # Define bounds
    size = ndepth * 2
    beta_mag_bounds = [(beta_mag_min, beta_mag_max)] * size
    beta_arg_bounds = [(beta_arg_min, beta_arg_max)] * size
    theta_bounds = [(theta_min, theta_max)] * size
    phi_bounds = [(phi_min, phi_max)] * size
    bounds = beta_mag_bounds + beta_arg_bounds + theta_bounds + phi_bounds

    # Random Initialization
    if len(Xvec) == 0:
        beta_mag = np.random.uniform(0, 3, size=(ndepth, 2))
        beta_arg = np.random.uniform(0, np.pi, size=(ndepth, 2))
        theta = np.random.uniform(0, np.pi, size=(ndepth, 2))
        phi = np.random.uniform(0, np.pi, size=(ndepth, 2))
        Xvec = bosonic_vqe_new.pack_variables(beta_mag, beta_arg, theta, phi)
        
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[0]))),name = 'qumode')
    qmr1 = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[1]))),name = 'qmr')
    qbr = QuantumRegister(1,name = 'qbit')
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1,qmr, qbr)

    beta_mag, beta_arg, theta, phi = bosonic_vqe_new.unpack_variables(Xvec, ndepth)
    circuit = bosonic_vqe_new.ecd_rot_ansatz(beta_mag, beta_arg, theta, phi, nfocks,circuit,qmr,qmr1,qbr)
    
    return circuit

def cv_qaoa(cutoff,s,a,p,n,maxiter=100,method = 'SLSQP'):
    costval = []
    estval = []
    
    def cost_function(params):
        return bosonic_qaoa.cvQAOA(params, cutoff, p, s, n, a, costval, estval)
    
    initial_params = np.random.uniform(0, 2 * np.pi, size=2 * p)

    result = minimize(
        cost_function,
        initial_params,   
        method=method,
        tol=1e-6,
        options={'maxiter': 100}
    )
    
    return result
    
def cv_qaoa_circuit(params,cutoff,s,a,p,n):
    gamma_list = params[:p] 
    eta_list = params[p:]  

    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),name= 'qumode')
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, cr)

    circuit.cv_initialize(0, qmr[0])       # vacuum
    circuit.cv_sq(-s, qmr[0])              # squeezing gate

    # QAOA unitaries
    for i in range(p):
        costH = bosonic_qaoa.cost(cutoff, a, n, eta_list[i])
        cost_gate = UnitaryGate(costH.full(), label=f'Uc_{eta_list[i]}')
        circuit.append(cost_gate, qmr[0])

        mixH = bosonic_qaoa.kinetic_mixer(cutoff, gamma_list[i])
        mixer_gate = UnitaryGate(mixH.full(), label=f'Um_{gamma_list[i]}')
        circuit.append(mixer_gate, qmr[0])
        
    return circuit

def shors_circuit(N, m, R, a, delta, cutoff):
    qmr1 = c2qa.QumodeRegister(num_qumodes=3, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1, qbr, cr)
    
    circuit = gkp_state_circuit(cutoff,circuit,qbr,qmr1,i=0)
    circuit = gkp_state_circuit(cutoff,circuit,qbr,qmr1,i=1)
    circuit.cv_sq(-np.log(delta), qmr1[2])
    
    circuit = shors_bq.translation_R(cutoff,R,circuit,qmr1,0)
    circuit = shors_bq.multiplication(cutoff,N,circuit,qmr1,1)
    circuit = shors_bq.U_aNm(cutoff, circuit, qmr1, qbr, a, N, m)

    return circuit


def state_transfer_CVtoDV_qft(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(1,n+1):
        V_j = state_transfer.Vj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        W_j = state_transfer.Wj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction
        
    return circuit

def state_transfer_DVtoCV_qft(cutoff,circuit,qmr,qbr,cr,n,lmbda=0.29):
    for j in range(n,0,-1):
        W_j = state_transfer.Wj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in x direction 
        
        V_j = state_transfer.Vj(lmbda,j,n,cutoff)
        gate1 = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate1, qmr[:] + qbr[:])  # adding custom gate : Conditional displacement in p direction
        
    return circuit

def qft_circuit(cutoff,delta, n, a, append):
    total = n + a + append
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode')
    qbr1 = QuantumRegister(n, name='qbits')
    append_reg = QuantumRegister(append, name='append')
    ancilla_reg = QuantumRegister(a, name='ancilla')
    creg = ClassicalRegister(n, name='creg') 
    
    circuit = c2qa.CVCircuit(qmr, ancilla_reg, qbr1, append_reg, creg)

    for q in ancilla_reg:
        circuit.h(q)

    total_reg = ancilla_reg[:] + qbr1[:] + append_reg[:]  # MSB to LSB
    reversed_reg = list(reversed(total_reg))             

    circuit.z(reversed_reg[0])
    circuit.z(reversed_reg[-1])

    circuit.h(reversed_reg[-1])

    for q in reversed_reg[:-1]:
        circuit.x(q)
        circuit.h(q)

    state_transfer_DVtoCV_qft(cutoff, circuit, qmr, total_reg, creg, total)

    delta_prime = (2*np.pi) / (2**total * delta)
    circuit.cv_d(delta / 2, qmr[0])
    circuit.cv_r(np.pi / 2, qmr[0])
    circuit.cv_d(-delta_prime / 2, qmr[0])

    state_transfer_CVtoDV_qft(cutoff, circuit, qmr, total_reg, creg, total)

    for q in reversed_reg[:-1]:
        circuit.h(q)
        circuit.x(q)
    circuit.h(reversed_reg[-1])
    circuit.z(reversed_reg[0])
    circuit.z(reversed_reg[-1])

    start_index = a  
    for i in range(n):
        circuit.measure(total_reg[start_index + i], creg[i])

    return circuit