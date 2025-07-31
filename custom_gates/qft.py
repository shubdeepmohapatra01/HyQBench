import numpy as np
from math import pi, ceil
import scipy
from qutip import *

def F(cutoff):
    return (1j*np.pi/2*num(cutoff)).expm()
