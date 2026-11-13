"""ADAPT-VQE for a saved DSRG effective Hamiltonian."""

from __future__ import annotations

from os import PathLike
from typing import Optional


def adapt_vqe(
    dsrg_hamiltonian: str | PathLike[str],
    *,
    adapt_it: int,
    active_electrons: int,
    device_name: Optional[str] = None,
    shots: Optional[int] = None,
    commutator_shots: Optional[int] = None,
    commutator_mode: str = "ansatz",
    commutator_debug: bool = False,
    pool_sample_size: Optional[int] = None,
    pool_seed: Optional[int] = None,
    optimizer_method: str = "BFGS",
    optimizer_maxiter: int = 100_000_000,
):
    """Run ADAPT-VQE for a DSRG effective Hamiltonian.

    The core ADAPT loop selects operators from a singles+doubles pool based on
    commutator magnitude, then optimizes the ansatz parameters at each
    iteration.

    Parameters
    ----------
    dsrg_hamiltonian
        Path to an NPZ file containing ``E0``, ``h1``, and ``h2`` from a
        nonrelativistic Forte DSRG calculation. The orbital and qubit counts
        are inferred from the tensor dimensions.
    adapt_it
        Number of ADAPT iterations.
    active_electrons
        Number of active electrons in the DSRG Hamiltonian.
    device_name
        PennyLane device to use. When omitted, prefer ``lightning.qubit`` and
        fall back to ``default.qubit``.
    shots
        If provided and > 0, run with shot-based sampling on the chosen device.
    commutator_shots
        If provided, override the shot count for commutator evaluations.
    commutator_mode
        ``"ansatz"`` uses the ansatz circuit to evaluate commutators; ``"statevec"``
        prepares the current statevector via ``qml.StatePrep`` before measuring.
    commutator_debug
        If True, compute both commutator modes per operator and report the
        maximum absolute difference per ADAPT iteration.
    pool_sample_size
        If provided, randomly sample this many operators from the pool per
        ADAPT iteration to reduce commutator evaluations.
    pool_seed
        Seed for the operator-pool sampler.
    optimizer_method
        SciPy optimization method (e.g. ``"BFGS"``, ``"COBYLA"``, ``"Nelder-Mead"``).
    optimizer_maxiter
        Maximum number of optimizer iterations per ADAPT iteration.

    Returns
    -------
    tuple
        ``(params, ash_excitation, energies)`` as produced by the optimization.

    Raises
    ------
    ValueError
        If the saved tensors or calculation options are inconsistent.
    ImportError
        If required optional dependencies are not installed.
    """

    if shots is not None and shots < 0:
        raise ValueError("shots must be >= 0")
    if commutator_shots is not None and commutator_shots < 0:
        raise ValueError("commutator_shots must be >= 0")
    if commutator_mode not in {"ansatz", "statevec"}:
        raise ValueError("commutator_mode must be 'ansatz' or 'statevec'")
    if pool_sample_size is not None and pool_sample_size <= 0:
        raise ValueError("pool_sample_size must be > 0")

    try:
        import re
        import warnings

        import numpy as np
        import pennylane as qml
        from pennylane import numpy as pnp
        from scipy.optimize import minimize

        try:
            from .hamiltonian import make_pennylane_hamiltonian
        except ImportError:  # pragma: no cover - direct script import
            from hamiltonian import make_pennylane_hamiltonian

        warnings.filterwarnings("ignore")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "adapt_vqe requires numpy, scipy, and pennylane (and optionally "
            "pennylane-lightning for a faster device)."
        ) from exc

    def _make_device(name: Optional[str], wires: int, device_shots: Optional[int]):
        kwargs = {}
        if device_shots is not None and device_shots > 0:
            kwargs["shots"] = device_shots
        if name is not None:
            return qml.device(name, wires=wires, **kwargs)
        # Backwards-compatible preference for lightning if available.
        try:
            return qml.device("lightning.qubit", wires=wires, **kwargs)
        except Exception:
            return qml.device("default.qubit", wires=wires, **kwargs)

    H, qubits, _ = make_pennylane_hamiltonian(
        qml, dsrg_hamiltonian, active_electrons=active_electrons
    )

    energies = []
    ash_excitation = []

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    comm_shots = shots if commutator_shots is None else commutator_shots
    if commutator_mode == "statevec" and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_mode='statevec' requires analytic commutator_shots")
    if commutator_debug and comm_shots is not None and comm_shots > 0:
        raise ValueError("commutator_debug requires analytic commutator_shots")
    dev_comm = _make_device(device_name, qubits, comm_shots)
    dev = _make_device(device_name, qubits, shots)
    dev_state = None
    dev_comm_state = None

    def _apply_ansatz(hf_state, ash_excitation, params):
        qml.BasisState(hf_state, wires=range(qubits))
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(
                    weight=params[i],
                    wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                    wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)),
                )
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(
                    weight=params[i],
                    wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)),
                )

    @qml.qnode(dev_comm)
    def commutator_expectation(params, ash_excitation, hf_state, H, w):
        _apply_ansatz(hf_state, ash_excitation, params)
        res = qml.commutator(H, w)
        return qml.expval(res)

    if commutator_mode == "statevec" or commutator_debug:
        dev_state = _make_device(device_name, qubits, None)
        dev_comm_state = _make_device(device_name, qubits, None)

        @qml.qnode(dev_state)
        def current_state(params, ash_excitation, hf_state):
            _apply_ansatz(hf_state, ash_excitation, params)
            return qml.state()

        @qml.qnode(dev_comm_state)
        def commutator_expectation_state(state, H, w):
            qml.StatePrep(state, wires=range(qubits))
            res = qml.commutator(H, w)
            return qml.expval(res)

    @qml.qnode(dev)
    def ash(params, ash_excitation, hf_state, H):
        _apply_ansatz(hf_state, ash_excitation, params)
        return qml.expval(H)

    def cost(params):
        return float(np.real(ash(params, ash_excitation, hf_state, H)))

    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    op1 = [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 = [
        qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})
        for x in doubles
    ]
    operator_pool = op1 + op2
    operator_pool_ops = [qml.fermi.jordan_wigner(op) for op in operator_pool]
    params = pnp.zeros(len(ash_excitation), requires_grad=True)
    rng = np.random.default_rng(pool_seed)

    for j in range(adapt_it):
        #print("The adapt iteration now is", j, flush=True)
        max_value = float("-inf")
        max_operator = None
        max_diff = 0.0
        state_for_comm = None
        if commutator_mode == "statevec" or commutator_debug:
            state_for_comm = current_state(params, ash_excitation, hf_state)
        if pool_sample_size is None or pool_sample_size >= len(operator_pool_ops):
            candidate_indices = range(len(operator_pool_ops))
        else:
            candidate_indices = rng.choice(
                len(operator_pool_ops), size=pool_sample_size, replace=False
            )

        for idx in candidate_indices:
            w = operator_pool_ops[idx]
            if commutator_mode == "statevec":
                exp_used = commutator_expectation_state(state_for_comm, H, w)
            else:
                exp_used = commutator_expectation(params, ash_excitation, hf_state, H, w)
            current_value = abs(2 * exp_used)
            if commutator_debug:
                if commutator_mode == "statevec":
                    exp_other = commutator_expectation(params, ash_excitation, hf_state, H, w)
                else:
                    exp_other = commutator_expectation_state(state_for_comm, H, w)
                max_diff = max(max_diff, abs(exp_used - exp_other))

            if current_value > max_value:
                max_value = current_value
                max_operator = operator_pool[idx]

        indices_str = re.findall(r"\d+", str(max_operator))
        excitations = [int(index) for index in indices_str]
        ash_excitation.append(excitations)

        params = np.append(np.asarray(params), 0.0)
        result = minimize(
            cost,
            params,
            method=optimizer_method,
            tol=1e-12,
            options={"disp": False, "maxiter": int(optimizer_maxiter)},
        )

        energies.append(result.fun)
        params = result.x
        #print("Energies are", energies, flush=True)
        if commutator_debug:
            print(f"Max commutator diff: {max_diff:.6e}", flush=True)
    print("gr_energy:", energies[-1])
    return params, ash_excitation, energies
