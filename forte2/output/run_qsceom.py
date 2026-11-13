"""Run ADAPT-VQE and QSC-EOM with the saved N2 DSRG Hamiltonian."""

import sys
from pathlib import Path

import numpy as np


# Load the standalone QSC-EOM modules without importing the compiled forte2
# package. This permits running in any environment with the dependencies below.
OUTPUT_DIR = Path(__file__).resolve().parent
QSCEOM_DIR = OUTPUT_DIR.parent / "qsceom"
sys.path.insert(0, str(QSCEOM_DIR))

from adaptvqe import adapt_vqe  # noqa: E402
from qsceom import qsceom  # noqa: E402


HAMILTONIAN_FILE = OUTPUT_DIR / "n2" / "n2_DSRG_Heff.npz"
OUTPUT_FILE = OUTPUT_DIR / "n2" / "n2_QSCEOM_results.txt"
ACTIVE_ELECTRONS = 6
ADAPT_ITERATIONS = 10
HARTREE_TO_EV = 27.211386245988


def main():
    """Optimize the reference and print QSC-EOM excited-state energies."""

    params, selected_excitations, vqe_energies = adapt_vqe(
        adapt_it=ADAPT_ITERATIONS,
        active_electrons=ACTIVE_ELECTRONS,
        dsrg_hamiltonian=HAMILTONIAN_FILE,
    )
    ground_energy = float(vqe_energies[-1])

    eom_results = qsceom(
        active_electrons=ACTIVE_ELECTRONS,
        params=params,
        ash_excitation=selected_excitations,
        shots=0,
        dsrg_hamiltonian=HAMILTONIAN_FILE,
    )
    excited_energies = np.real_if_close(np.asarray(eom_results[0]))
    if np.iscomplexobj(excited_energies):
        raise RuntimeError("QSC-EOM returned eigenvalues with non-negligible imaginary parts")

    output_lines = [
        f"Ground-state energy: {ground_energy:.12f} Ha",
        "QSC-EOM excited states:",
    ]
    for state, energy in enumerate(excited_energies, start=1):
        energy = float(energy)
        gap = energy - ground_energy
        output_lines.append(
            f"State {state:2d}: E = {energy:.12f} Ha  "
            #f"Delta E = {gap:.12f} Eh ({gap * HARTREE_TO_EV:.8f} eV)"
        )

    output_text = "\n".join(output_lines) + "\n"
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")
    print("\n" + output_text, end="")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
