"""Check the N2 example's output path and active-electron metadata."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("existing_directory", [False, True])
def test_n2_export_creates_output_directory(tmp_path, monkeypatch, existing_directory):
    output_dir = tmp_path / "n2"
    if existing_directory:
        output_dir.mkdir()

    seen = {}

    class FakeSystem:
        def __init__(self, **kwargs):
            assert output_dir.is_dir()  # Created before the calculation starts.
            self.__dict__.update(kwargs)
            seen["system"] = self

    class FakeState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeRHF:
        def __init__(self, **kwargs):
            pass

        def __call__(self, system):
            return self

    class FakeCISolver:
        def __init__(self, states, core_orbitals, active_orbitals, nroots):
            seen["state"] = states
            seen["core_orbitals"] = core_orbitals
            seen["active_orbitals"] = active_orbitals
            seen["nroots"] = nroots
            self.sub_solvers = [SimpleNamespace(state=states, ncore=core_orbitals)]

    class FakeMCOptimizer:
        def __init__(self, ci_solver):
            self.ci_solver = ci_solver

        def __call__(self, rhf):
            return self

    class FakeDSRG:
        def __init__(self, flow_param, relax_reference):
            self.flow_param = flow_param
            self.hbar0 = -1.0
            self.hbar1_canon = np.eye(6)
            self.hbar2_canon = np.zeros((6, 6, 6, 6))
            self.relax_eigvals = np.array([-1.5, -1.0, -0.5])

        def __call__(self, mc):
            self.ci_solver = mc.ci_solver
            return self

        def run(self):
            return self

    fake_forte = ModuleType("forte2")
    fake_forte.System = FakeSystem
    fake_forte.State = FakeState
    fake_forte.RHF = FakeRHF
    fake_forte.CISolver = FakeCISolver
    fake_forte.MCOptimizer = FakeMCOptimizer
    fake_dsrg = ModuleType("forte2.dsrg")
    fake_dsrg.DSRG_MRPT2 = FakeDSRG
    monkeypatch.setitem(sys.modules, "forte2", fake_forte)
    monkeypatch.setitem(sys.modules, "forte2.dsrg", fake_dsrg)

    script = Path(__file__).resolve().parents[2] / "forte2" / "output" / "run_dsrg.py"
    namespace = {"__file__": str(tmp_path / "run_dsrg.py"), "__name__": "__main__"}
    exec(compile(script.read_text(), str(script), "exec"), namespace)

    assert seen["state"].nel == 14
    assert seen["core_orbitals"] == 4
    assert seen["active_orbitals"] == 6
    assert seen["nroots"] == 3
    with np.load(output_dir / "n2_DSRG_Heff.npz") as data:
        assert data["active_electrons"] == 6
        assert data["h1"].shape == (6, 6)
        assert data["geometry_unit"] == "angstrom"
