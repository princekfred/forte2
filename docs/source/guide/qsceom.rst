ADAPT-VQE and QSC-EOM with a DSRG Hamiltonian
============================================

ADAPT-VQE and QSC-EOM read a saved nonrelativistic DSRG effective Hamiltonian.
The NPZ file must contain ``E0``, ``h1``, and ``h2``. The number of spatial
orbitals is inferred from ``h1``; specify the number of active electrons
separately. If the file contains ``active_electrons`` metadata, it must match
the supplied count.

Both calculations must use the same file and active-electron count:

.. code-block:: python

    from forte2.qsceom.adaptvqe import adapt_vqe
    from forte2.qsceom.qsceom import qsceom

    heff = "H4_DSRG_Heff.npz"
    params, excitations, energies = adapt_vqe(
        heff, adapt_it=10, active_electrons=4
    )
    eom_roots = qsceom(
        heff,
        active_electrons=4,
        params=params,
        ash_excitation=excitations,
    )

The spatial two-body tensor is converted from Forte's convention to
PennyLane's convention before the Jordan--Wigner mapping. The saved scalar
term is included in the qubit Hamiltonian. These algorithms require NumPy,
SciPy, and PennyLane; PySCF is not required for the saved-Hamiltonian path.

For the repository's H4 example, ``python forte2/output/run_qsceom.py``
loads the already-generated NPZ directly and can run in a PennyLane
environment without importing Forte2's compiled extension.
