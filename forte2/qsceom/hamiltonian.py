"""Utilities for loading DSRG effective Hamiltonians."""

from __future__ import annotations

from os import PathLike

import numpy as np


def load_dsrg_hamiltonian(
    path: str | PathLike[str], *, active_electrons: int | None = None
):
    """Load and validate a DSRG effective Hamiltonian saved by ``run_dsrg.py``.

    Forte stores ``h2[p,q,r,s] = (pr|qs)``, whereas PennyLane's
    ``fermionic_observable`` consumes ``two[p,q,r,s] = (ps|qr)``. The last
    two axes are therefore exchanged when constructing the PennyLane operator.

    Parameters
    ----------
    path
        NPZ file containing the keys ``E0``, ``h1``, and ``h2``.
    active_electrons
        If provided, check the electron count against the orbital count and,
        when present, the ``active_electrons`` metadata in the NPZ file.

    Returns
    -------
    tuple
        ``(constant, one_body, two_body, n_orbitals)``.  ``constant`` has
        shape ``(1,)`` as required by PennyLane.
    """

    with np.load(path) as data:
        missing = {"E0", "h1", "h2"} - set(data.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"DSRG Hamiltonian file is missing: {names}")
        constant = np.asarray(data["E0"]).reshape(-1)
        one_body = np.asarray(data["h1"])
        two_body = np.asarray(data["h2"])
        saved_electrons = (
            np.asarray(data["active_electrons"]).item()
            if "active_electrons" in data
            else None
        )

    if constant.size != 1:
        raise ValueError("E0 must be a scalar")
    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("h1 must be a square rank-2 tensor")

    n_orbitals = one_body.shape[0]
    if n_orbitals == 0:
        raise ValueError("h1 must contain at least one spatial orbital")
    expected_h2_shape = (n_orbitals,) * 4
    if two_body.shape != expected_h2_shape:
        raise ValueError(f"h2 must have shape {expected_h2_shape}, got {two_body.shape}")
    if active_electrons is not None:
        if not isinstance(active_electrons, (int, np.integer)) or not (
            0 <= active_electrons <= 2 * n_orbitals
        ):
            raise ValueError(
                f"active_electrons must be an integer between 0 and {2 * n_orbitals}"
            )
        if saved_electrons is not None and active_electrons != saved_electrons:
            raise ValueError(
                f"active_electrons={active_electrons} does not match the DSRG "
                f"Hamiltonian metadata ({saved_electrons})"
            )

    return constant, one_body, two_body, n_orbitals


def make_pennylane_hamiltonian(
    qml, path, cutoff: float = 1.0e-20, *, active_electrons: int | None = None
):
    """Convert a saved nonrelativistic DSRG Hamiltonian to a qubit operator.

    Forte stores ``V[p,q,r,s]`` with the operator ordering
    ``a+(p) a+(q) a(s) a(r)``. PennyLane uses ``a+(p) a+(q) a(r) a(s)``,
    so the last two tensor indices must be swapped. PennyLane supplies the
    two-body factor of one half and includes the scalar term.
    """

    constant, one_body, two_body, n_orbitals = load_dsrg_hamiltonian(
        path, active_electrons=active_electrons
    )
    pennylane_two_body = two_body.swapaxes(2, 3)
    fermionic_hamiltonian = qml.qchem.fermionic_observable(
        constant, one_body, pennylane_two_body, cutoff=cutoff
    )
    return qml.jordan_wigner(fermionic_hamiltonian), 2 * n_orbitals, n_orbitals
