from pathlib import Path

import numpy as np
from forte2 import System, RHF, State, CISolver, MCOptimizer
from forte2.dsrg import DSRG_MRPT2

output_file = Path(__file__).resolve().parent / "n2" / "n2_DSRG_Heff.npz"
output_file.parent.mkdir(parents=True, exist_ok=True)

# 1. Define molecule and basis set
system = System(
    xyz="""
    N 0.0 0.0 0.0
    N 0.0 0.0 3.0
    """,
    unit="angstrom",
    basis_set="cc-pVDZ",
    auxiliary_basis_set="cc-pVTZ-JKFIT",
)

# 2. Hartree-Fock reference
rhf = RHF(charge=0)(system)

# 3. State-averaged active-space calculation
ci_solver = CISolver(
    states=State(nel=14, multiplicity=1, ms=0.0),
    core_orbitals=4,
    active_orbitals=6,
    nroots=3,
)

mcscf = MCOptimizer(ci_solver=ci_solver)(rhf)

# 4. DSRG with reference relaxation
dsrg = DSRG_MRPT2(
    flow_param=0.5,
    relax_reference="once",
)(mcscf)

dsrg.run()

# 5. Extract the effective Hamiltonian
E0 = dsrg.hbar0
h1 = dsrg.hbar1_canon
h2 = dsrg.hbar2_canon


# 6. Save results for subsequent VQE calculations
sub_solver = dsrg.ci_solver.sub_solvers[0]
np.savez(
    output_file,
    E0=E0,
    h1=h1,
    h2=h2,
    active_electrons=sub_solver.state.nel - 2 * sub_solver.ncore,
    relaxed_state_energies=dsrg.relax_eigvals,
    geometry_xyz=system.xyz,
    geometry_unit=system.unit,
    basis_set=system.basis_set,
    auxiliary_basis_set=system.auxiliary_basis_set,
    flow_param=dsrg.flow_param,
)

print("Effective Hamiltonian saved to:", output_file)
print("Scalar term:", E0)
print("One-body tensor shape:", h1.shape)
print("Two-body tensor shape:", h2.shape)
print("Relaxed state energies:", dsrg.relax_eigvals)
