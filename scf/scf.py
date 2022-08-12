#!/usr/bin/env python3


import jax
import jax.numpy as jnp


N = 3 # Basis set degree (i.e. STO-NG)
R = 1.4632 # Bond length in Angstrom
ZETA1 = 2.0925 # Orbital exponent (atom 1, He)
ZETA2 = 1.24 # Orbital exponent (atom 2, H)
ZA = 2.0 # Nuclear charge (atom 1, He)
ZB = 1.0 # Nuclear charge (atom 2, H)

MAX_ITER = 15 # Maximum number of SCF iterations

# Basis function coefficients for STO-NG (N = 1--3)
COEFFICIENTS = jnp.array([[1.000000, 0.000000, 0.000000],
                          [0.678914, 0.430129, 0.000000],
                          [0.444635, 0.535328, 0.154329]])

# Basis function exponents for STO-NG (N = 1--3)
EXPONENTS = jnp.array([[0.270950, 0.000000, 0.00000],
                       [0.151623, 0.851819, 0.00000],
                       [0.109818, 0.405771, 2.22766]])


def overlap_int(a, b, r):
    """Compute electron overlap integrals.

    Args:
        a (float): _description_
        b (float): _description_
        r (float): _description_

    Returns:
        float: _description_
    """
    return jnp.power(jnp.pi / (a + b), 1.5) * jnp.exp((-1 * a * b * r) / (a + b))


def kinetic_energy_int(a, b, r):
    """Compute kinetic energy integrals.

    Args:
        a (float): _description_
        b (float): _description_
        r (float): _description_

    Returns:
        float: _description_
    """
    return (a * b) / (a + b) * (3 - (2 * a * b * r) / (a + b)) * overlap_int(a, b, r)


def nuclear_int(a, b, r1, r2, z):
    """Compute nuclear-electron attraction integrals.

    Args:
        a (float): _description_
        b (float): _description_
        r1 (float): _description_
        r2 (float): _description_
        z (float): _description_

    Returns:
        float: _description_
    """
    return -1 * z * 2 * jnp.pi / (a + b) * f0((a + b) * r2) * jnp.exp(-1 * a * b * r1 / (a + b))


def f0(x):
    """Compute zeroth order Boys function.

    Args:
        x (float): _description_

    Returns:
        float: _description_
    """
    return jnp.where(x < 1e-6,
                     1.0 - x/3.0,
                     jnp.sqrt(jnp.pi / x) * jax.scipy.special.erf(jnp.sqrt(x)) / 2)


def two_electron_int(a, b, c, d, r1, r2, r3):
    """Compute two-electron repulsion integrals.

    Args:
        a (float): _description_
        b (float): _description_
        c (float): _description_
        d (float): _description_
        r1 (float): _description_
        r2 (float): _description_
        r3 (float): _description_

    Returns:
        float: _description_
    """
    return 2 * (jnp.power(jnp.pi, 2.5) / ((a + b) * (c + d) * jnp.sqrt(a + b + c + d))
                * f0((a + b) * (c + d) * r3 / (a + b + c + d))
                * jnp.exp(-1 * a * b * r1 / (a + b) - c * d * r2 / (c + d)))


def scale_basis():
    """Scale the STO basis functions according to nuclear charge.

    Returns:
        DeviceArray: _description_
        DeviceArray: _description_
    """
    return (EXPONENTS[N-1, :] * jnp.power(ZETA1, 2), # A1
            EXPONENTS[N-1, :] * jnp.power(ZETA2, 2)) # A2


def normalise_basis(A1, A2):
    """Normalise the cartesian basis functions.

    Args:
        A1 (DeviceArray): _description_
        A2 (DeviceArray): _description_

    Returns:
        DeviceArray: _description_
        DeviceArray: _description_
    """
    return (COEFFICIENTS[N-1, :] * jnp.power(2 * A1 / jnp.pi, 0.75), # D1
            COEFFICIENTS[N-1, :] * jnp.power(2 * A2 / jnp.pi, 0.75)) # D2


def calc_overlap_integrals(A1, A2, D1, D2):
    return jnp.sum(overlap_int(A1, A2[:, jnp.newaxis], R*R) * jnp.einsum('i,j', D1, D2))


def calc_kinetic_energy_integrals(A1, A2, D1, D2):
    T11 = jnp.sum(kinetic_energy_int(A1, A1[:, None], 0.0) * jnp.einsum('i,j', D1, D1))
    T12 = jnp.sum(kinetic_energy_int(A1, A2[:, None], R*R) * jnp.einsum('i,j', D1, D2))
    T22 = jnp.sum(kinetic_energy_int(A2, A2[:, None], 0.0) * jnp.einsum('i,j', D2, D2))
    return (T11, T12, T22)


def calc_nuclear_repulsion_integrals(A1, A2, D1, D2):
    rAP = A2[:, None] * R / (A1 + A2[:, None])
    rBP = R - rAP
    V11a = jnp.sum(nuclear_int(A1, A1[:, None],   0,       0, ZA) * jnp.einsum('i,j', D1, D1))
    V12a = jnp.sum(nuclear_int(A1, A2[:, None], R*R, rAP*rAP, ZA) * jnp.einsum('i,j', D1, D2))
    V22a = jnp.sum(nuclear_int(A2, A2[:, None],   0,     R*R, ZA) * jnp.einsum('i,j', D2, D2))
    V11b = jnp.sum(nuclear_int(A1, A1[:, None],   0,     R*R, ZB) * jnp.einsum('i,j', D1, D1))
    V12b = jnp.sum(nuclear_int(A1, A2[:, None], R*R, rBP*rBP, ZB) * jnp.einsum('i,j', D1, D2))
    V22b = jnp.sum(nuclear_int(A2, A2[:, None],   0,       0, ZB) * jnp.einsum('i,j', D2, D2))
    return (V11a, V12a, V22a, V11b, V12b, V22b)


def calc_two_electron_integrals(A1, A2, D1, D2):
    rAP = A2 * R / (A2 + A1[:, None])
    rAQ = A2[:, None, None] * R / (A2[:, None, None] + A1[:, None, None, None])
    rPQ = rAP - rAQ
    rBQ = R - rAQ
    V1111 = jnp.sum(two_electron_int(A1, A1[:, None], A1[:, None, None], A1[:, None, None, None],   0,   0,       0)
                    * jnp.einsum('i,j,k,l->ijkl', D1, D1, D1, D1))
    V2111 = jnp.sum(two_electron_int(A2, A1[:, None], A1[:, None, None], A1[:, None, None, None], R*R,   0, rAP*rAP)
                    * jnp.einsum('i,j,k,l->ijkl', D2, D1, D1, D1))
    V2121 = jnp.sum(two_electron_int(A2, A1[:, None], A2[:, None, None], A1[:, None, None, None], R*R, R*R, rPQ*rPQ)
                    * jnp.einsum('i,j,k,l->ijkl', D2, D1, D2, D1))
    V2211 = jnp.sum(two_electron_int(A2, A2[:, None], A1[:, None, None], A1[:, None, None, None],   0,   0,     R*R)
                    * jnp.einsum('i,j,k,l->ijkl', D2, D2, D1, D1))
    V2221 = jnp.sum(two_electron_int(A2, A2[:, None], A2[:, None, None], A1[:, None, None, None],   0, R*R, rBQ*rBQ)
                    * jnp.einsum('i,j,k,l->ijkl', D2, D2, D2, D1))
    V2222 = jnp.sum(two_electron_int(A2, A2[:, None], A2[:, None, None], A2[:, None, None, None],   0,   0,       0)
                    * jnp.einsum('i,j,k,l->ijkl', D2, D2, D2, D2))
    return (V1111, V2111, V2121, V2211, V2221, V2222)


def compute_integrals():
    # Prepare the basis set
    A1, A2 = scale_basis()
    D1, D2 = normalise_basis(A1, A2)

    # Compute all the integrals
    S12 = calc_overlap_integrals(A1, A2, D1, D2)
    T11, T12, T22 = calc_kinetic_energy_integrals(A1, A2, D1, D2)
    V11a, V12a, V22a, V11b, V12b, V22b = calc_nuclear_repulsion_integrals(A1, A2, D1, D2)
    V1111, V2111, V2121, V2211, V2221, V2222 = calc_two_electron_integrals(A1, A2, D1, D2)

    # Build the core Hamiltonian
    H0 = jnp.array([[T11 + V11a + V11b, T12 + V12a + V12b],
                    [T12 + V12a + V12b, T22 + V22a + V22b]])

    # The overlap matrix
    S = jnp.array([[  1, S12],
                   [S12,   1]])

    # Orthogonalisation matrix
    X = jnp.array([[1 / jnp.sqrt(2 * (1 + S12)),  1 / jnp.sqrt(2 * (1 - S12))],
                   [1 / jnp.sqrt(2 * (1 + S12)), -1 / jnp.sqrt(2 * (1 - S12))]])

    # Two-electron integrals. I assume I should do this with a dictionary or
    # hash map but following Szabo and Ostlund will do for now.
    TT = jnp.zeros((2, 2, 2, 2))
    TT = TT.at[0, 0, 0, 0].set(V1111)
    TT = TT.at[1, 0, 0, 0].set(V2111)
    TT = TT.at[0, 1, 0, 0].set(V2111)
    TT = TT.at[0, 0, 1, 0].set(V2111)
    TT = TT.at[0, 0, 0, 1].set(V2111)
    TT = TT.at[1, 0, 1, 0].set(V2121)
    TT = TT.at[0, 1, 1, 0].set(V2121)
    TT = TT.at[1, 0, 0, 1].set(V2121)
    TT = TT.at[0, 1, 0, 1].set(V2121)
    TT = TT.at[1, 1, 0, 0].set(V2211)
    TT = TT.at[0, 0, 1, 1].set(V2211)
    TT = TT.at[1, 1, 1, 0].set(V2221)
    TT = TT.at[1, 1, 0, 1].set(V2221)
    TT = TT.at[1, 0, 1, 1].set(V2221)
    TT = TT.at[0, 1, 1, 1].set(V2221)
    TT = TT.at[1, 1, 1, 1].set(V2222)

    return (H0, S, X, TT)


def G(p, tt):
    return jnp.einsum('kl,ijkl->ij', p, tt) - jnp.einsum('kl,ilkj->ij', p, 0.5 * tt)


def main():
    # Compute core hamiltonian H0, overlap integrals S, orthogonalisation
    # transformation matrix X, and two-electron integrals TT.
    H0, S, X, TT = compute_integrals()

    # Initialise the density matrix to zero
    P = jnp.zeros((2, 2))

    # SCF iterations begin here
    for n_iter in range(MAX_ITER):
        # Two-electron part of the Fock matrix
        g = G(P, TT)

        # Fock = Core + two-electron
        F = H0 + g
    
        # Electronic energy
        e_E = jnp.sum(0.5 * P * (H0 + F))

        # Solve Roothan-Hall equation FC = SCe for the expansion coeffients C.
        # First, need to orthogonalise the Fock matrix:
        Fp = X.T @ F @ X

        # The eigenvectors Cp of the orthogonalised Fock matrix Fp are related
        # to the expansion coefficients C by Cp = X^{-1} C
        _, Cp = jnp.linalg.eig(Fp.T)
        C = X @ Cp

        # Calculate (change in) density matrix
        old_P = P.copy()
        P = 2 * jnp.einsum('i,j', C[:, 0], C[:, 0])

        # If the change in density is small enough, end iterating
        delta = jnp.sqrt(jnp.power(P - old_P, 2).sum() / 4)
        if delta < 1e-6:
            break

    # Total energy is electronic + nuclear
    TOTAL_E = e_E + ZA * ZB / R
    print(TOTAL_E)


if __name__ == '__main__':
    main()