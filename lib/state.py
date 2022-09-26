from __future__ import annotations
import functools
import numpy as np
import numpy.typing as npt


def normalize_phi(phi: npt.NDArray[np.complex128]):
    norm = np.linalg.norm(phi)
    if norm == 0:
        raise ValueError("Wavefunction is not normalizable.")

    return (phi / norm).astype(np.complex128)


def standardize_phi(phi: npt.NDArray[np.complex128]):
    phi = normalize_phi(phi)
    max = np.abs(phi).max()

    for value in phi:
        if np.abs(value) > 0.01 * max:
            phase_factor = value / np.abs(value)
            return phi / phase_factor
    return phi


@functools.cache
def create_basis_state(k: int, N: int):
    ket = np.zeros(N, dtype=np.complex128)
    ket[k] = 1
    return ket

