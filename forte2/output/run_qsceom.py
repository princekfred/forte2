"""Run ADAPT-VQE and QSC-EOM with the saved H4 DSRG Hamiltonian."""

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np


# Load the standalone QSC-EOM modules without importing the compiled forte2
# package. This permits running in any environment with the dependencies below.
OUTPUT_DIR = Path(__file__).resolve().parent
QSCEOM_DIR = OUTPUT_DIR.parent / "qsceom"
sys.path.insert(0, str(QSCEOM_DIR))

from adaptvqe import adapt_vqe  # noqa: E402
from qsceom import qsceom  # noqa: E402


HAMILTONIAN_FILE = OUTPUT_DIR / "h4" / "H4_DSRG_Heff.npz"
OUTPUT_FILE = OUTPUT_DIR / "h4" / "H4_QSCEOM_results.txt"
ACTIVE_ELECTRONS = 4
ADAPT_ITERATIONS = 10
HARTREE_TO_EV = 27.211386245988


def main():
    """Optimize the reference and print QSC-EOM excited-state energies."""

    # Use the same bytes for ADAPT and QSC-EOM even if DSRG regenerates the
    # original NPZ while the calculation is running.
    hamiltonian_bytes = HAMILTONIAN_FILE.read_bytes()
    fingerprint = hashlib.sha256(hamiltonian_bytes).hexdigest()
    print(f"Hamiltonian: {HAMILTONIAN_FILE.resolve()}", flush=True)
    print(f"Hamiltonian SHA-256: {fingerprint}", flush=True)

    with tempfile.TemporaryDirectory(prefix="qsceom_hamiltonian_") as directory:
        snapshot = Path(directory) / "hamiltonian.npz"
        snapshot.write_bytes(hamiltonian_bytes)
        with np.load(snapshot) as data:
            active_orbitals = data["h1"].shape[0]
            reference_energies = (
                np.asarray(data["relaxed_state_energies"]).reshape(-1).copy()
                if "relaxed_state_energies" in data
                else None
            )

        params, selected_excitations, vqe_energies = adapt_vqe(
            adapt_it=ADAPT_ITERATIONS,
            active_electrons=ACTIVE_ELECTRONS,
            dsrg_hamiltonian=snapshot,
        )
        ground_energy = float(vqe_energies[-1])

        eom_results = qsceom(
            active_electrons=ACTIVE_ELECTRONS,
            params=params,
            ash_excitation=selected_excitations,
            shots=0,
            dsrg_hamiltonian=snapshot,
        )
    excited_energies = np.real_if_close(np.asarray(eom_results[0]))
    if np.iscomplexobj(excited_energies):
        raise RuntimeError("QSC-EOM returned eigenvalues with non-negligible imaginary parts")

    output_lines = [
        f"Hamiltonian: {HAMILTONIAN_FILE.resolve()}",
        f"Hamiltonian SHA-256: {fingerprint}",
        f"Active electrons: {ACTIVE_ELECTRONS}",
        f"Active spatial orbitals: {active_orbitals}",
        f"ADAPT iterations: {len(vqe_energies)}",
        f"Ground-state energy: {ground_energy:.12f} Ha",
    ]
    if reference_energies is not None and reference_energies.size:
        reference_energy = float(np.min(np.real_if_close(reference_energies)))
        output_lines.extend(
            [
                f"Lowest relaxed DSRG root: {reference_energy:.12f} Ha",
                f"ADAPT minus relaxed DSRG root: {ground_energy - reference_energy:.12f} Ha",
            ]
        )
    output_lines.append("QSC-EOM excited states:")
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
