import c2qa
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RGate
from qiskit.converters import circuit_to_gate
from qutip import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate

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

def CD_real(cutoff,alpha):
    p = momentum(cutoff)
    cdReal = tensor(sigmax(),-1j*np.sqrt(2)*alpha*p).expm()
    return cdReal

def CD_imaginary(cutoff,alpha):
    x = position(cutoff)
    cdImaginary = tensor(sigmay(),-1j*np.pi*x/(4*alpha)).expm()
    return cdImaginary

def Ux_operator(cutoff,theta,alpha,delta):
    CD_real = tensor(sigmax(),1j*(theta/alpha)*position(cutoff)).expm()
    CD_imag = tensor(sigmay(),1j*(2*theta/alpha)*(delta**2)*momentum(cutoff)).expm()
    return CD_real * CD_imag

def conditional_displacement(cutoff,alpha):
    D_plus = displace(cutoff, alpha)
    D_minus = displace(cutoff, -alpha)
    return tensor(qproj00(),D_plus) + tensor(qproj11(),D_minus)