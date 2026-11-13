"""Tests for the DSRG-only ADAPT-VQE and QSC-EOM entry points."""

import builtins
import hashlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


qml = pytest.importorskip("pennylane")
QSCEOM_DIR = Path(__file__).resolve().parents[2] / "forte2" / "qsceom"


def _load_module(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", QSCEOM_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def algorithms(monkeypatch):
    # The user can run these modules without importing forte2's C++ extension.
    monkeypatch.syspath_prepend(str(QSCEOM_DIR))
    return _load_module("adaptvqe").adapt_vqe, _load_module("qsceom").qsceom


def test_only_dsrg_hamiltonian_source_is_exposed(algorithms):
    for algorithm in algorithms:
        parameters = inspect.signature(algorithm).parameters
        assert parameters["dsrg_hamiltonian"].default is inspect.Parameter.empty
        assert not {
            "symbols",
            "geometry",
            "basis",
            "charge",
            "spin",
            "active_orbitals",
            "hamiltonian_cutoff",
        } & parameters.keys()


@pytest.mark.parametrize("scalar", [0.7, 1.1])
def test_algorithms_read_saved_hamiltonian(tmp_path, monkeypatch, algorithms, scalar):
    adapt_vqe, qsceom = algorithms
    path = tmp_path / "dsrg.npz"
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = 0.4
    h2[1, 1, 1, 1] = 0.5
    h2[0, 1, 0, 1] = h2[1, 0, 1, 0] = 0.25
    np.savez(
        path,
        E0=scalar,
        h1=np.diag([-1.0, -0.3]),
        h2=h2,
        active_electrons=2,
    )

    original_import = builtins.__import__

    def no_pyscf(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("The DSRG path must not import PySCF")
        return original_import(name, *args, **kwargs)

    def no_molecular_hamiltonian(*args, **kwargs):
        raise AssertionError("The DSRG path must not build a molecular Hamiltonian")

    monkeypatch.setattr(builtins, "__import__", no_pyscf)
    monkeypatch.setattr(qml.qchem, "molecular_hamiltonian", no_molecular_hamiltonian)

    params, excitations, energies = adapt_vqe(
        path, adapt_it=1, active_electrons=2, device_name="default.qubit"
    )
    assert energies[-1] == pytest.approx(scalar - 1.6, abs=1.0e-9)

    eigenvalues = qsceom(
        path, active_electrons=2, params=params, ash_excitation=excitations
    )[0]
    np.testing.assert_allclose(
        eigenvalues, [scalar - 1.05, scalar - 1.05, scalar - 0.1], atol=1.0e-9
    )


def test_active_electron_count_checked_against_file(tmp_path, algorithms):
    adapt_vqe, qsceom = algorithms
    path = tmp_path / "dsrg.npz"
    np.savez(
        path,
        E0=0.0,
        h1=np.eye(2),
        h2=np.zeros((2, 2, 2, 2)),
        active_electrons=2,
    )
    with pytest.raises(ValueError, match="does not match"):
        adapt_vqe(path, adapt_it=1, active_electrons=1)
    with pytest.raises(ValueError, match="does not match"):
        qsceom(path, active_electrons=1, params=[], ash_excitation=[])


def test_runner_passes_same_hamiltonian_snapshot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(QSCEOM_DIR))
    runners = sorted(QSCEOM_DIR.parent.glob("**/run_qsceom.py"))
    if not runners:
        pytest.skip("The optional H4 example runner is not present")
    runner_path = runners[0]
    spec = importlib.util.spec_from_file_location("_test_qsceom_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    source = tmp_path / "dsrg.npz"
    np.savez(
        source,
        E0=0.7,
        h1=np.diag([-1.0, -0.3]),
        h2=np.zeros((2, 2, 2, 2)),
        active_electrons=2,
        relaxed_state_energies=[-1.3],
    )
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    runner.HAMILTONIAN_FILE = source
    runner.OUTPUT_FILE = tmp_path / "results.txt"
    runner.ACTIVE_ELECTRONS = 2
    runner.ADAPT_ITERATIONS = 1

    def fake_adapt(*, dsrg_hamiltonian, adapt_it, active_electrons):
        assert adapt_it == 1 and active_electrons == 2
        with np.load(dsrg_hamiltonian) as data:
            assert float(data["E0"]) == pytest.approx(0.7)
        # Simulate DSRG replacing the original while ADAPT is running.
        np.savez(source, E0=99.0, h1=np.eye(2), h2=np.zeros((2, 2, 2, 2)))
        return np.array([]), [], [-1.3]

    def fake_qsceom(*, dsrg_hamiltonian, active_electrons, params, ash_excitation, shots):
        assert active_electrons == 2 and shots == 0
        with np.load(dsrg_hamiltonian) as data:
            assert float(data["E0"]) == pytest.approx(0.7)
        return [np.array([-0.6])]

    monkeypatch.setattr(runner, "adapt_vqe", fake_adapt)
    monkeypatch.setattr(runner, "qsceom", fake_qsceom)
    runner.main()
    result = runner.OUTPUT_FILE.read_text()
    assert fingerprint in result
    assert "Active spatial orbitals: 2" in result
    assert "Lowest relaxed DSRG root: -1.300000000000" in result
