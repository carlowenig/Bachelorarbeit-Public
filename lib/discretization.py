import functools
import numpy as np

from .op import BasisFunctionOp, FunctionOp, commutator
from .state import create_basis_state

hbar: float = 1
d: float = 1


# --- DESCRETIZATION OPERATOR ---


def k_vals(N):
    return np.arange(N, dtype=np.complex128)


k_op = BasisFunctionOp(lambda k, N: create_basis_state(k, N) * k)
k_op.validate_dagger()


# --- POSITION OPERATOR ---


def calc_dx(N):
    return complex(2.0 * d / N)


def x_vals(N):
    return -d + calc_dx(N) * (k_vals(N) + 1 / 2)


# Implement dx as an operator, since it dependends on N.
dx_op = FunctionOp(lambda phi: calc_dx(len(phi)) * phi)
dx_op.validate_dagger()


X_op = -d + dx_op * (k_op + 1 / 2)
X_op.validate_dagger()
assert X_op.is_hermitian()


# --- WAVEFUNCTION ---


def compute_C(psi_vals):
    return 1 / np.sqrt(np.sum(np.abs(psi_vals) ** 2))


def psi_to_phi(N, psi):
    psi_vals = np.array([psi(x) for x in x_vals(N)])
    phi = compute_C(psi_vals) * psi_vals

    assert np.isclose(np.sum(np.abs(phi) ** 2), 1), "Phi is not normalized."

    return phi


# --- TRANSLATION OPERATOR ---


@functools.cache
def create_T_op(R=1):
    return BasisFunctionOp(
        lambda k, N: create_basis_state((k - 1), N)
        if k > 0
        else create_basis_state(N - 1, N) * R,
        lambda k, N: create_basis_state((k + 1), N)
        if k < N - 1
        else create_basis_state(0, N) * R,
    )


T_op_0 = create_T_op(0)
T_op_0.validate_dagger()
assert not T_op_0.is_unitary()

T_op_1 = create_T_op(1)
T_op_1.validate_dagger()
assert T_op_1.is_unitary()


# --- DIFFERENTIATION OPERATOR ---

recip_dx_op = FunctionOp(lambda phi: 1 / calc_dx(len(phi)) * phi)


D_methods = ["V", "R", "Z", "5P"]


def D_method_label(method):
    return "$\\hat D_\\mathrm{" + method + "}$"


@functools.cache
def create_D_op(method, R=1):
    T_op = create_T_op(R=R)
    if method == "V":
        return (T_op - 1) * recip_dx_op
    elif method == "R":
        return (1 - T_op.dagger) * recip_dx_op
    elif method == "Z":
        return (T_op - T_op.dagger) / 2 * recip_dx_op
    elif method == "5P":
        return (
            (-T_op * T_op + 8 * T_op - 8 * T_op.dagger + T_op.dagger * T_op.dagger)
            / 12
            * recip_dx_op
        )
    else:
        raise ValueError("Unknown method.")


# --- MOMENTUM OPERATOR ---


@functools.cache
def create_P_op(D_method, R=1):
    return -1j * hbar * create_D_op(D_method, R=R)


assert create_P_op("Z").is_hermitian()


# --- HAMILTONIAN OPERATOR ---


@functools.cache
def create_H0_op(method, R=1, m=1):
    if method in ("RV", "VR"):
        p_squ = create_P_op("R", R=R) * create_P_op("V", R=R)
    elif method == "Z":
        p_squ = create_P_op("Z", R=R) ** 2
    elif method == "5P":
        p_squ = create_P_op("5P", R=R) ** 2
    else:
        raise ValueError(f"Unknown method {method}.")

    return p_squ / (2 * m)


H0_methods = ["VR", "Z", "5P"]


@functools.cache
def create_H_op(H0_method, V, R=1, m=1):
    return create_H0_op(H0_method, R=R, m=m) + V


test_H_op = create_H_op("Z", X_op * X_op)
assert test_H_op.is_hermitian()
assert test_H_op.is_real()


# --- DELTA OPERATOR ---


@functools.cache
def create_delta_op(D_method, R=1):
    return 1 + commutator(X_op, create_D_op(D_method, R=R))


# --- COHERENT STATES ---


@functools.cache
def create_coherent_psi(q=0, p=0, m=1, omega=100):
    def psi(x):
        c = (m * omega / (np.pi * hbar)) ** 0.25
        return c * np.exp(1j / hbar * p * x - (x - q) ** 2 * m * omega / (2 * hbar))

    return psi


@functools.cache
def create_coherent_phi(N, **kwargs):
    return psi_to_phi(N, create_coherent_psi(**kwargs))
