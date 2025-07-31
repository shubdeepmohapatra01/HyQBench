import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qiskit.quantum_info import partial_trace
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from scipy.stats.contingency import margins

def qproj00():
    return basis(2, 0).proj()


def qproj11():
    return basis(2, 1).proj()


def qproj01():
    op = np.array([[0, 1], [0, 0]])
    return Qobj(op)


def qproj10():
    op = np.array([[0, 0], [1, 0]])
    return Qobj(op)

def hadamard():
    op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return Qobj(op)

def Q_displacement_plus1(cutoff):
    return (-1j*momentum(cutoff)).expm()
def Q_displacement_minus1(cutoff):
    return (1j*momentum(cutoff)).expm()
def bosonic_sum(cutoff1,cutoff2):
    return (-1j*tensor(position(cutoff1),momentum(cutoff2))).expm()
def single_mode_squeeze(cutoff,squeeze_param):
    return (1j*squeeze_param*(position(cutoff)*momentum(cutoff) + momentum(cutoff)*position(cutoff))/2).expm()
def Q_control_plus1(cutoff):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(-1j*momentum(cutoff)).expm())
def Q_control_minus1(cutoff):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*momentum(cutoff)).expm())
def P_displacement_pi(cutoff,sign):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi*sign*position(cutoff)).expm())
def rotation_control(cutoff,sign):
    return tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi/2*sign*num(cutoff)).expm())

def multiplication(cutoff,alpha,circuit,qumode_register,i):
    if alpha == 1:
        return circuit

    log_alpha = np.log(alpha)
    l = int(np.ceil(abs(log_alpha)))
    if l == 0:  
        small_r = 0
    else:
        small_r = -log_alpha / l

    for i in range(l):
        circuit.cv_sq(small_r,qumode_register[i])

    return circuit

def extractLSB(cutoff,circuit,qumode_register,qubit_register,i):
    qumode_gate = tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(1j*np.pi*position(cutoff)).expm())
    gate1 =UnitaryGate(qumode_gate.full(), label='LSB')
    
    circuit.h(qubit_register[0])
    circuit.append(gate1, qumode_register[i]+qubit_register[:])
    circuit.h(qubit_register[0])
    
    return circuit

def extractLSB_dag(cutoff,circuit,qumode_register,qubit_register,i):
    qumode_gate = tensor(qproj00(),qeye(cutoff)) + tensor(qproj11(),(-1j*np.pi*position(cutoff)).expm())
    gate1 =UnitaryGate(qumode_gate.full(), label='LSB_dag')
    
    circuit.h(qubit_register[0])
    circuit.append(gate1, qumode_register[i]+qubit_register[:])
    circuit.h(qubit_register[0])
    
    return circuit

def translation_R(cutoff,R,circuit,qumode_register,i):
    circuit.cv_d(R/np.sqrt(2),qumode_register[i])
    return circuit

def control_multiplication(cutoff,alpha,circuit,qumode_register,qubit_register,i):
    rotation_plus = rotation_control(cutoff,1)
    gate1 = UnitaryGate(rotation_plus.full(), label='cR1')
 
    rotation_minus = rotation_control(cutoff,-1)
    gate2 = UnitaryGate(rotation_minus.full(), label='cR2')
    
    circuit = multiplication(cutoff,np.sqrt(alpha),circuit,qumode_register,i)
    circuit.append(gate2, qumode_register[i]+qubit_register[:])
    circuit = multiplication(cutoff,1/np.sqrt(alpha),circuit,qumode_register,i)
    circuit.append(gate1, qumode_register[i]+qubit_register[:])
    
    
    return circuit

def V_alpha(cutoff,circuit,qumode_register,qubit_register,alpha):
    circuit = multiplication(cutoff,2,circuit,qumode_register,2)
    
    circuit = extractLSB(cutoff,circuit,qumode_register,qubit_register,0)
    
    control_subtraction = Q_control_minus1(cutoff)
    gate1 =UnitaryGate(control_subtraction.full(), label='Q-1')
    circuit.append(gate1, qumode_register[0]+qubit_register[:])
    
    circuit = control_multiplication(cutoff,alpha,circuit,qumode_register,qubit_register,1)
    
    control_addition = Q_control_plus1(cutoff)
    gate1 =UnitaryGate(control_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[2]+qubit_register[:])
    
    circuit = extractLSB(cutoff,circuit,qumode_register,qubit_register,2)
    
    circuit = multiplication(cutoff,0.5,circuit,qumode_register,0)

    return circuit

def V_alpha_dag(cutoff,circuit,qumode_register,qubit_register,alpha):
    circuit = multiplication(cutoff, 2,circuit,qumode_register,0)

    circuit = extractLSB_dag(cutoff,circuit,qumode_register,qubit_register,2)

    control_subtraction = Q_control_minus1(cutoff)
    gate1 = UnitaryGate(control_subtraction.full(), label='Q-1')
    circuit.append(gate1, qumode_register[2] + qubit_register[:])

    circuit = control_multiplication(cutoff, 1/alpha,circuit,qumode_register,qubit_register,1)

    control_addition = Q_control_plus1(cutoff)
    gate1 = UnitaryGate(control_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[0] + qubit_register[:])

    circuit = extractLSB_dag(cutoff,circuit,qumode_register,qubit_register,0)

    circuit = multiplication(cutoff, 0.5,circuit,qumode_register,2)
    
    return circuit

def V_aNm(cutoff,circuit,qumode_register,qubit_register,a,N,m):
    for i in range(m):
        alpha = pow(a,2**i) % N
        circuit = V_alpha(cutoff,circuit,qumode_register,qubit_register,alpha)
        circuit.barrier()

    for i in range(m):
        circuit = V_alpha_dag(cutoff,circuit,qumode_register,qubit_register,1)
        circuit.barrier()
        
    return circuit

def V_aNm_dagger(cutoff, circuit, qumode_register, qubit_register, a, N, m):
    for _ in range(m):
        V_alpha(cutoff, circuit, qumode_register, qubit_register, 1)
        circuit.barrier()

    for i in reversed(range(m)):
        alpha = pow(a, 2**i, N)
        V_alpha_dag(cutoff, circuit, qumode_register, qubit_register, alpha)
        circuit.barrier()
    
    return circuit

def U_aNm(cutoff, circuit, qumode_register, qubit_register, a, N, m):
    circuit = V_aNm_dagger(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()
    circuit.barrier()
    
    Q_addition = Q_displacement_plus1(cutoff)
    gate1 = UnitaryGate(Q_addition.full(), label='Q+1')
    circuit.append(gate1, qumode_register[1])
    circuit.barrier()
    circuit.barrier()
    
    circuit = V_aNm(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()
    circuit.barrier()
    
    return circuit
     

def position_plotting(state, cutoff, ax_min=-6, ax_max=6, steps=500):
    x = position(cutoff)
    expval = expect(x, Qobj(state))


    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_dist, _ = margins(w.T)  # Marginalize over y-axis

    x_dist *= (ax_max - ax_min) / steps
    xaxis = np.linspace(ax_min, ax_max, steps)
    return x_dist, xaxis

def momentum_plotting(state, cutoff, ax_min=-6, ax_max=6, steps=500):
    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    _, p_dist = margins(w.T)  # Marginalize over x-axis for momentum

    p_dist *= (ax_max - ax_min) / steps
    paxis = np.linspace(ax_min, ax_max, steps)
    return p_dist, paxis

def trace_out_qumode_index(circuit,state,qumode_register,qubit_register,qumode_index='0'):
    if(qumode_index == '0'):
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[1]+qumode_register[2])
    elif(qumode_index == '1'):
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0]+qumode_register[2])
    else:
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0]+qumode_register[1])
        
    return trace

def get_reduced_qumode_density_matrix(stateop, qumode_index, num_qumodes, cutoff):
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    total_qubits = num_qumodes * num_qubits_per_qumode + 1

    keep_indices = list(range(
        qumode_index * num_qubits_per_qumode,
        (qumode_index + 1) * num_qubits_per_qumode
    ))

    all_indices = list(range(total_qubits))
    trace_indices = [i for i in all_indices if i not in keep_indices]

    return partial_trace(stateop, trace_indices)