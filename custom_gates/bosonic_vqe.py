import c2qa
import numpy as np
import sympy as sp
import qutip as qt
import qiskit
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor
from qiskit import QuantumRegister
from qiskit.circuit.library import UnitaryGate
from functools import partial
from scipy.optimize import minimize

def qproj00():
    return qt.basis(2, 0).proj()


def qproj11():
    return qt.basis(2, 1).proj()


def qproj01():
    op = np.array([[0, 1], [0, 0]])
    return qt.Qobj(op)


def qproj10():
    op = np.array([[0, 0], [1, 0]])
    return qt.Qobj(op)


def pack_variables(beta_mag, beta_arg, theta, phi):
    Xvec = np.concatenate([
        beta_mag.ravel(),
        beta_arg.ravel(),
        theta.ravel(),
        phi.ravel()
    ])
    return Xvec


def unpack_variables(Xvec, ndepth):
    size = ndepth * 2

    beta_mag = Xvec[:size].reshape((ndepth, 2))
    beta_arg = Xvec[size:2*size].reshape((ndepth, 2))
    theta = Xvec[2*size:3*size].reshape((ndepth, 2))
    phi = Xvec[3*size:].reshape((ndepth, 2))

    return beta_mag, beta_arg, theta, phi

def get_cvec_np(r, theta):
    r = np.array(r)
    theta = np.array(theta)
    return r * np.exp(1j * theta)

def qubit_rot(theta, phi):
    """
    R (theta, phi) = exp[ −i (theta/2) ( X cos(phi) + Y sin(phi) ) ].

    Arguments:
    theta, phi: rotation parameters
    """
    gen = ( qt.sigmax() * np.cos(phi) )
    gen += ( qt.sigmay() * np.sin(phi) )

    H = -1j * (theta / 2) * gen

    return H.expm()

def ecd_op(beta, theta, phi, cind, nfocks):
    """
    ECD operator.

    Arguments:
    beta -- ECD parameter
    theta, phi -- rotation parameters
    cind -- qumode index
    nfocks -- Fock cutoffs
    """
    # Validate cind
    if cind not in (0, 1):
        raise ValueError("cind must be 0 or 1")

    # ECD
    if cind == 0:
        E2 = qt.tensor(qproj10(), qt.displace(nfocks[0], beta / 2))
        E2 += qt.tensor(qproj01(), qt.displace(nfocks[0], -beta / 2))
        E2 = qt.tensor(E2, qt.qeye(nfocks[1]))
    else:
        E2 = qt.tensor(qproj10(), qt.qeye(nfocks[0]), qt.displace(nfocks[1], beta / 2))
        E2 += qt.tensor(qproj01(), qt.qeye(nfocks[0]), qt.displace(nfocks[1], -beta / 2))

    return E2

def ecd_rot_op(beta, theta, phi, nfocks):
    """
    ECD-rotation operator.

    Arguments:
    beta -- ECD parameters
    theta, phi -- rotation parameters
    nfocks -- Fock cutoffs
    """
    # Rotations
    R1 = qt.tensor(qubit_rot(theta[0], phi[0]), qt.qeye(nfocks[0]), qt.qeye(nfocks[1]))
    R2 = qt.tensor(qubit_rot(theta[1], phi[1]), qt.qeye(nfocks[0]), qt.qeye(nfocks[1]))

    # ECDs
    E1 = ecd_op(beta[0], theta[0], phi[0], 0, nfocks)
    E2 = ecd_op(beta[1], theta[1], phi[1], 1, nfocks)

    return E2 * R2 * E1 * R1

def ecd_rot_ansatz(bmag_mat, barg_mat, theta_mat, phi_mat, nfocks):
    """
    ECD-rotation ansatz.

    Arguments:
    bmag_mat, barg_mat -- ECD parameters
    theta_mat, phi_mat -- rotation parameters
    nfocks -- Fock cutoffs
    """
    # Check
    if bmag_mat.shape != barg_mat.shape:
        raise ValueError("Dimensions of bmag_mat and barg_mat do not match.")
    beta_mat = get_cvec_np(bmag_mat, barg_mat)
    if beta_mat.shape != theta_mat.shape:
        raise ValueError("Lengths of beta_mat and theta_mat do not match.")
    if beta_mat.shape != phi_mat.shape:
        raise ValueError("Lengths of beta_mat and phi_mat do not match.")

    # Initialize
    ndepth = beta_mat.shape[0]
    uni = ecd_rot_op(beta_mat[0, :], theta_mat[0, :], phi_mat[0, :], nfocks)

    # Check
    if ndepth == 1:
        return uni

    # Loop through blocks
    for i in range(1, ndepth):
        new_uni = ecd_rot_op(beta_mat[i, :], theta_mat[i, :], phi_mat[i, :], nfocks)
        uni = ( new_uni * uni )

    return uni

def binary_knapsack_ham(l_val, values, weights, max_weight, include_id=False):
    """
    Generates the binary Hamiltonian for the knapsack problem.

    Arguments:
    l_val -- lambda penalty parameter
    values -- item values
    weights -- item weights
    max_weight -- total weight capacity
    include_id -- identity as symbol (True) or value 1 (False)

    Returns:
    H_total -- The binary Hamiltonian
    """
    # Number of primary variables
    N_qb = len(weights)

    # Symbols
    symbol_list = list(sp.symbols('x_:{}'.format(str(N_qb))))
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.

    # E = sum(i) v(i) x(i)
    H_prob = sum(values[ii] * symbol_list[ii] for ii in range(N_qb))

    # Calculate scaling factor
    max_weight_bin_str = bin(max_weight).lstrip('0b')  # Step 1
    max_val = 2**len(max_weight_bin_str) - 1  # Step 2
    scaling_factor = max_val / max_weight  # Step 3

    # Apply scaling to weights
    scaled_weights = [weight * scaling_factor for weight in weights]

    # W0 = sum(i) w(i) x(i)
    H_constraints = sum(scaled_weights[ii] * symbol_list[ii] for ii in range(N_qb))

    # Bitstring representation converted to list
    bin_weight = list(bin(max_weight).lstrip('0b'))[::-1]

    # Auxiliary symbol indices start after the primary variables
    aux_symbols = sp.symbols('x_{}:{}'.format(str(N_qb), str(N_qb + len(bin_weight))))

    # A = sum(i) 2^i y(i)
    H_constraints_aux = sum(aux_symbols[ii] * 2**ii for ii in range(len(bin_weight)))

    # Full Hamiltonian
    H_total = -H_prob + l_val * (max_weight - H_constraints - H_constraints_aux)**2

    # Construct full list of symbols in expression
    symbol_list.extend(list(aux_symbols))

    # Create a list of x_j**2
    sq_syms = [temp_sym**2 for temp_sym in symbol_list]

    # Maps x_j^{2} to x_j:
    conv_dict = dict(zip(sq_syms, list(symbol_list)))

    # Final binary Hamiltonian
    H_total = H_total.subs(conv_dict)

    return H_total, symbol_list

def binary_to_qubit_ham(H_bin, symbol_list, include_id=False):
    """
    Map a symbolic binary Hamiltonian to a spin Hamiltonian.

    Arguments:
    H_bin -- The binary Hamiltonian as a SymPy object
    symbol_list -- SymPy symbols defining the Hamiltonian
    include_id -- identity as symbol (True) or value 1 (False)
    """
    # Initialize spin variables (Z0, Z1, ..., Zn)
    z_symbols = sp.symbols('z:{}'.format(len(symbol_list)))

    # Define the identity operator (I_j)
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.0

    # Create a mapping dictionary from binary symbols to spin expressions
    bin2spin_dict = {
        symbol: (1/2)*(Ident - z) for symbol, z in zip(symbol_list, z_symbols)
    }

    # Convert the binary Hamiltonian to a spin Hamiltonian
    spin_ham = H_bin.subs(bin2spin_dict).expand()

    # Z^2 = I
    sq_z = [z**2 for z in z_symbols]
    sq_values = [Ident] * len(z_symbols)  # All squared terms map to the identity
    spin_squared_map = dict(zip(sq_z, sq_values))

    # Substitute squared terms
    red_spin_ham = spin_ham.subs(spin_squared_map)

    return red_spin_ham


def check_spinz(input_list, spinz):
    out_val = ['I']*len(spinz)
    for ll in range(len(input_list)):
        out_val[int(input_list[ll].strip('z'))] = 'Z'
    return out_val


def sympy_to_pauli_dict(smpy_exp):
    """
    Convert a sympy spin Hamiltonian expression to a dictionary with
    Pauli words as keys and string coefficients as values.
    """
    # Determine the number of qubits
    spinz = smpy_exp.free_symbols

    # Split at spaces so we have the individual terms/coefficients
    split_expr = str(smpy_exp).split()

    # Firs iteration
    matrix_dict = {}
    split_term = split_expr[0].split('*')
    tmp_coeff = split_term[0]
    tmp_paulis = split_term[1:]
    pauli_word = ''.join(check_spinz(tmp_paulis, spinz))
    matrix_dict[pauli_word] = tmp_coeff

    # Iterate through the remaining terms
    for ii in range(1, len(split_expr), 2):
        tmp_sign = split_expr[ii]
        split_term = split_expr[ii+1].split('*')
        tmp_coeff  = split_term[0]
        tmp_paulis = split_term[1:]
        pauli_word = ''.join(check_spinz(tmp_paulis, spinz))
        matrix_dict[pauli_word] = tmp_sign+tmp_coeff

    return matrix_dict


def binary_to_pauli_list(H_total, symbol_list):
    """
    Maps a binary Hamiltonian to Pauli terms and coefficients.

    Arguments:
    H_total -- The binary Hamiltonian
    symbol_list -- symbols defining the Hamiltonian
    """
    spin_ham = binary_to_qubit_ham(H_total, symbol_list)
    op_dict = sympy_to_pauli_dict(spin_ham)

    return [[key, float(value)] for key, value in op_dict.items()]

def generate_tensor_product(string):
    """
    Get QuTip object given a string representing a Pauli word.
    """
    # Define a mapping of characters to corresponding QuTiP operators
    operator_map = {
        'I': qt.qeye(2),  # Identity operator
        'X': qt.sigmax(),  # Pauli-X operator
        'Y': qt.sigmay(),  # Pauli-Y operator
        'Z': qt.sigmaz()   # Pauli-Z operator
    }

    # Create a list to collect the operators
    operators = []

    # Append the corresponding operators based on the input string
    for char in string:
        operators.append(operator_map[char])

    # Compute the tensor product of all operators in the list
    U = qt.tensor(*operators).full()

    return qt.Qobj(U)


def qubit_op_to_ham(pterms):
    """
    Get QuTip object given a set of Pauli words and correspdoing coefficients.
    """
    terms = []
    for p in pterms:
        term = ( p[1] * generate_tensor_product(p[0]) )
        terms.append(term)

    return sum(terms)

def gate_from_ecd(Xvec, ndepth, nfocks):
    """
    Qumode state |Psi> = U ( |0> |0, 0> ).

    Arguments:
    Xvec -- ECD-rotation parameters
    ndepth -- circuit depth
    nfocks -- Fock cutoffs
    """
    # Parameters
    beta_mag, beta_arg, theta, phi = unpack_variables(Xvec, ndepth)

    # ECD unitary
    U = ecd_rot_ansatz(beta_mag, beta_arg, theta, phi, nfocks)
    # print(type(U))

    # U |0, 0, 0>
    # vac = qt.tensor( qt.basis(2, 0), qt.basis(nfocks[0], 0), qt.basis(nfocks[1], 0) )
    # psi = U * vac

    return U

def energy_val(Xvec, ndepth, nfocks, H):
    """
    Compute <psi | H |psi > where

    |psi (n, m)> <== U ( |0> |0, 0> ).

    Arguments:
    Xvec -- ansatz ECD-rotation parameters
    H -- QuTip two-qumode Hamiltonian
    nfock -- Fock cutoff for qumode
    ndepth -- circuit depth
    """
    # Qubit-qubit-qumode state
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[0]))),name = 'qmr')
    qmr1 = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[1]))),name = 'qmr1')
    qbr = QuantumRegister(1,name = 'qbit')
    cr = qiskit.ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1,qmr, qbr)

    Ugate = gate_from_ecd(Xvec, ndepth, nfocks)
    # print(Ugate.full().shape)
    ecd1 = UnitaryGate(Ugate.full(), label=f'ECD')
    circuit.append(ecd1, qmr1[:] + qmr[:] + qbr[:])

    state, result, _ = c2qa.util.simulate(circuit)


    # Expectation value
    ham = qt.Qobj( H.full() )
    en = qt.expect(ham, qt.Qobj(state))

    return en

def ecd_opt_vqe(H, ndepth, nfocks, maxiter=100, method='COBYLA', verb=0,
                threshold=1e-08, print_freq=10, Xvec=[]):
    """
    Minimize the cost function using SciPy-based methods.

    Arguments:
    H -- QuTip Hamiltonian
    ndepth -- ansatz circuit depth
    nfocks -- Fock cutoffs
    maxiter -- maximum number of iterations
    method -- optimization method
    threshold -- error tolerance
    Xvec -- optional initial guesses
    print_freq -- frequency of printing and storing intermediate results
    """
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

    # Guess
    if len(Xvec) == 0:
        beta_mag = np.random.uniform(0, 3, size=(ndepth, 2))
        beta_arg = np.random.uniform(0, np.pi, size=(ndepth, 2))
        theta = np.random.uniform(0, np.pi, size=(ndepth, 2))
        phi = np.random.uniform(0, np.pi, size=(ndepth, 2))
        Xvec = pack_variables(beta_mag, beta_arg, theta, phi)

    # Loss function
    obj_fun = partial(energy_val, ndepth=ndepth, nfocks=nfocks, H=H)

    # Intermediate values
    iteration_step = 0
    intermediate_results = []

    def callback(xk):
        nonlocal iteration_step
        iteration_step += 1
        loss_value = obj_fun(xk)
        if verb == 1 and (iteration_step % print_freq == 0):
            print("-------------------")
            print(f"iter: {iteration_step}")
            print(f"fval: {loss_value}")

        # Store intermediate results
        if iteration_step % print_freq == 0:
            intermediate_results.append((loss_value, xk.copy()))

    # SciPy options
    options = {'disp': True, 'maxiter': maxiter}

    # Optimize
    result = minimize(obj_fun, Xvec, method=method, bounds=bounds,
                             tol=threshold, options=options, callback=callback)

    return result.fun, result.x, intermediate_results

def binary_knapsack_ham(l_val, values, weights, max_weight, include_id=False):
    """
    Generates the binary Hamiltonian for the knapsack problem.

    Arguments:
    l_val -- lambda penalty parameter
    values -- item values
    weights -- item weights
    max_weight -- total weight capacity
    include_id -- identity as symbol (True) or value 1 (False)

    Returns:
    H_total -- The binary Hamiltonian
    """
    # Number of primary variables
    N_qb = len(weights)

    # Symbols
    symbol_list = list(sp.symbols('x_:{}'.format(str(N_qb))))
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.

    # E = sum(i) v(i) x(i)
    H_prob = sum(values[ii] * symbol_list[ii] for ii in range(N_qb))

    # Calculate scaling factor
    max_weight_bin_str = bin(max_weight).lstrip('0b')  # Step 1
    max_val = 2**len(max_weight_bin_str) - 1  # Step 2
    scaling_factor = max_val / max_weight  # Step 3

    # Apply scaling to weights
    scaled_weights = [weight * scaling_factor for weight in weights]

    # W0 = sum(i) w(i) x(i)
    H_constraints = sum(scaled_weights[ii] * symbol_list[ii] for ii in range(N_qb))

    # Bitstring representation converted to list
    bin_weight = list(bin(max_weight).lstrip('0b'))[::-1]

    # Auxiliary symbol indices start after the primary variables
    aux_symbols = sp.symbols('x_{}:{}'.format(str(N_qb), str(N_qb + len(bin_weight))))

    # A = sum(i) 2^i y(i)
    H_constraints_aux = sum(aux_symbols[ii] * 2**ii for ii in range(len(bin_weight)))

    # Full Hamiltonian
    H_total = -H_prob + l_val * (max_weight - H_constraints - H_constraints_aux)**2

    # Construct full list of symbols in expression
    symbol_list.extend(list(aux_symbols))

    # Create a list of x_j**2
    sq_syms = [temp_sym**2 for temp_sym in symbol_list]

    # Maps x_j^{2} to x_j:
    conv_dict = dict(zip(sq_syms, list(symbol_list)))

    # Final binary Hamiltonian
    H_total = H_total.subs(conv_dict)

    return H_total, symbol_list

def generate_tensor_product(string):
    """
    Get QuTip object given a string representing a Pauli word.
    """
    # Define a mapping of characters to corresponding QuTiP operators
    operator_map = {
        'I': qeye(2),  # Identity operator
        'X': sigmax(),  # Pauli-X operator
        'Y': sigmay(),  # Pauli-Y operator
        'Z': sigmaz()   # Pauli-Z operator
    }

    # Create a list to collect the operators
    operators = []

    # Append the corresponding operators based on the input string
    for char in string:
        operators.append(operator_map[char])

    # Compute the tensor product of all operators in the list
    U = tensor(*operators).full()

    return Qobj(U)


def qubit_op_to_ham(pterms):
    """
    Get QuTip object given a set of Pauli words and correspdoing coefficients.
    """
    terms = []
    for p in pterms:
        term = ( p[1] * generate_tensor_product(p[0]) )
        terms.append(term)

    return sum(terms)

def binary_to_qubit_ham(H_bin, symbol_list, include_id=False):
    """
    Map a symbolic binary Hamiltonian to a spin Hamiltonian.

    Arguments:
    H_bin -- The binary Hamiltonian as a SymPy object
    symbol_list -- SymPy symbols defining the Hamiltonian
    include_id -- identity as symbol (True) or value 1 (False)
    """
    # Initialize spin variables (Z0, Z1, ..., Zn)
    z_symbols = sp.symbols('z:{}'.format(len(symbol_list)))

    # Define the identity operator (I_j)
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.0

    # Create a mapping dictionary from binary symbols to spin expressions
    bin2spin_dict = {
        symbol: (1/2)*(Ident - z) for symbol, z in zip(symbol_list, z_symbols)
    }

    # Convert the binary Hamiltonian to a spin Hamiltonian
    spin_ham = H_bin.subs(bin2spin_dict).expand()

    # Z^2 = I
    sq_z = [z**2 for z in z_symbols]
    sq_values = [Ident] * len(z_symbols)  # All squared terms map to the identity
    spin_squared_map = dict(zip(sq_z, sq_values))

    # Substitute squared terms
    red_spin_ham = spin_ham.subs(spin_squared_map)

    return red_spin_ham


def check_spinz(input_list, spinz):
    out_val = ['I']*len(spinz)
    for ll in range(len(input_list)):
        out_val[int(input_list[ll].strip('z'))] = 'Z'
    return out_val


def sympy_to_pauli_dict(smpy_exp):
    """
    Convert a sympy spin Hamiltonian expression to a dictionary with
    Pauli words as keys and string coefficients as values.
    """
    # Determine the number of qubits
    spinz = smpy_exp.free_symbols

    # Split at spaces so we have the individual terms/coefficients
    split_expr = str(smpy_exp).split()

    # Firs iteration
    matrix_dict = {}
    split_term = split_expr[0].split('*')
    tmp_coeff = split_term[0]
    tmp_paulis = split_term[1:]
    pauli_word = ''.join(check_spinz(tmp_paulis, spinz))
    matrix_dict[pauli_word] = tmp_coeff

    # Iterate through the remaining terms
    for ii in range(1, len(split_expr), 2):
        tmp_sign = split_expr[ii]
        split_term = split_expr[ii+1].split('*')
        tmp_coeff  = split_term[0]
        tmp_paulis = split_term[1:]
        pauli_word = ''.join(check_spinz(tmp_paulis, spinz))
        matrix_dict[pauli_word] = tmp_sign+tmp_coeff

    return matrix_dict


def binary_to_pauli_list(H_total, symbol_list):
    """
    Maps a binary Hamiltonian to Pauli terms and coefficients.

    Arguments:
    H_total -- The binary Hamiltonian
    symbol_list -- symbols defining the Hamiltonian
    """
    spin_ham = binary_to_qubit_ham(H_total, symbol_list)
    op_dict = sympy_to_pauli_dict(spin_ham)

    return [[key, float(value)] for key, value in op_dict.items()]

def generate_triples(nfocks):
    # Create ranges for q, n, and m
    q_range = np.arange(2)
    n_range = np.arange(nfocks[0])
    m_range = np.arange(nfocks[1])

    # Create a meshgrid of q, n, and m with valid indexing
    q_grid, n_grid, m_grid = np.meshgrid(q_range, n_range, m_range, indexing='ij')

    # Stack the grids to get (q, n, m) triples
    triples = np.stack((q_grid.ravel(), n_grid.ravel(), m_grid.ravel()), axis=-1)

    return triples

def num_prob_basis(Xvec, nvec, ndepth, nfocks):
    """
    | <psi | q, n, m> |^2, where

    |psi> <== U |0, 0, 0>.

    Arguments:
    Xvec -- ansatz parameters
    nvec -- Fock basis state indices
    nfocks -- Fock cutoffs
    ndepth -- circuit depth
    """
    # Qubit-qubit-qumode state
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[0]))),name = 'qmr')
    qmr1 = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[1]))),name = 'qmr1')
    qbr = QuantumRegister(1,name = 'qbit')
    cr = qiskit.ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr1,qmr, qbr)

    Ugate = gate_from_ecd(Xvec, ndepth, nfocks)
    # print(Ugate.full().shape)
    ecd1 = UnitaryGate(Ugate.full(), label=f'ECD')
    circuit.append(ecd1, qmr1[:] + qmr[:] + qbr[:])

    psi, result, _ = c2qa.util.simulate(circuit)

    # |q, n, m >
    state = qt.tensor(qt.basis(2, nvec[0]),
                      qt.basis(nfocks[0], nvec[1]),
                      qt.basis(nfocks[1], nvec[2]) )

    # Expectation value
    P0 = qt.Qobj(psi).overlap(state)

    return np.abs(P0)**2


def num_prob_all(Xvec, ndepth, nfocks):
    """
    | <psi | q, n, m> |^2 for all (n, m).

    Arguments:
    Xvec -- ansatz parameters
    nfock -- Fock cutoff for single qumode
    nvec -- Fock basis state indices
    """
    # Initialize
    N1 = generate_triples(nfocks)
    ntriples = N1.shape[0]

    # Generate
    P1 = []
    for i in range(ntriples):
        P1.append( num_prob_basis(Xvec, N1[i, :], ndepth, nfocks) )

    return np.array(P1)

