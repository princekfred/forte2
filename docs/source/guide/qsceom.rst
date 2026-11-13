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

The repository example ``python forte2/output/run_qsceom.py`` loads the NPZ
selected by ``HAMILTONIAN_FILE`` and can run in a PennyLane environment
without importing Forte2's compiled extension.

DSRG corrections and the saved Hamiltonian
------------------------------------------

``DSRG_MRPT2`` includes a second-order dynamical-correlation correction from
outside the active-space reference. The saved ``E0``, ``h1``, and ``h2``
are the scalar, one-body, and two-body parts of the resulting effective
Hamiltonian. ADAPT-VQE and QSC-EOM use that Hamiltonian directly, so the DSRG
correction must not be added to their energies a second time.

With ``relax_reference="once"``, Forte diagonalizes the effective Hamiltonian
once to obtain the relaxed roots stored as ``relaxed_state_energies``. For
the N2 example with a 3.0 angstrom bond length, cc-pVDZ, CAS(6e,6o), and flow
parameter 0.5, the saved file contained the following roots when this note
was written (hartree):

.. code-block:: text

    Root 0: -108.947666801431
    Root 1: -108.750810281720
    Root 2: -108.750662771489

These numbers are an example from ``forte2/output/n2/n2_DSRG_Heff.npz``;
regenerating the file with different inputs can change them. On 13 November
2026, the saved file predated the latest edit to ``run_dsrg.py``; regenerate
it after checking the input if results for the current settings are needed.

In the DSRG log, ``E0`` is the current reference energy and ``Edsrg`` is the
DSRG energy before reference relaxation. Their difference is the
state-averaged second-order correction for a state-averaged calculation. For
example, after ``dsrg.run()`` it can be printed with:

.. code-block:: python

    correction = dsrg.relax_energies[0, 0] - dsrg.relax_energies[0, 2]
    print("State-averaged DSRG correction:", correction, "hartree")

This difference is not an individual root's energy shift. The ``E0`` stored
in the NPZ is also different from the log's reference ``E0``: the stored
value is the scalar part of the effective Hamiltonian. The NPZ does not store
the state-averaged correction separately; retain the DSRG log to inspect it.
