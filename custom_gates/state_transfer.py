import numpy as np
from math import pi, ceil
import scipy
from qutip import *

def Vj(lmbda,j,n,cutoff):
    qubit_pauli = tensor([identity(2)]*(j-1) + [sigmay()] + [identity(2)]*(n-j))
    return tensor(qubit_pauli, (1j*np.pi)/(2**(j+1)*lmbda)*(destroy(cutoff) + create(cutoff))/np.sqrt(2.0)).expm()

def Wj(lmbda,j,n,cutoff): 
    qubit_pauli = tensor([identity(2)]*(j-1) + [sigmax()] + [identity(2)]*(n-j))
    if j == n:      
        disp_amount = -lmbda*2**(j-1)
    else:
        disp_amount = lmbda*2**(j-1)
    # print(qubit_pauli)
    return tensor(qubit_pauli, disp_amount*(destroy(cutoff) - create(cutoff))/np.sqrt(2.0)).expm()

def dv2cv_st_non_abelian(lmbda,n,cutoff):
    # return the unitary operator that does state transfer from DV to CV
    U = tensor([identity(2)]*n + [identity(cutoff)])
    for j in range(n,0,-1):
        U = Vj(lmbda,j,n,cutoff).dag() * Wj(lmbda,j,n,cutoff).dag() * U

    return U