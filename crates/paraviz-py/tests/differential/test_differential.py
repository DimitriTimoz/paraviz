"""Differential tests: run the same `paraview.simple` case scripts under real
ParaView (conda env, `PV_REAL_PYTHON`) and under the paraviz shim, then compare
the JSON traces (pipeline structure, property read-backs, Fetch data stats).

Skipped automatically when no real ParaView python is available.
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(HERE, "cases")
CASES = sorted(
    f[:-3] for f in os.listdir(CASES_DIR) if f.endswith(".py") and f != "__init__.py"
)

PV_REAL = os.environ.get(
    "PV_REAL_PYTHON",
    "/tmp/pvx/ParaView-6.0.1-MPI-OSX11.0-Python3.12-arm64/ParaView-6.0.1.app/Contents/bin/pvpython",
)
HAS_REAL = os.path.exists(PV_REAL)


def _run(python, case, out_json, workdir):
    os.makedirs(workdir, exist_ok=True)
    r = subprocess.run(
        [python, os.path.join(HERE, "run_case.py"), case, out_json],
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"case {case} failed under {python}:\n{r.stdout}\n{r.stderr}")
    with open(out_json) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    """Runs every case under both pythons once; returns {case: (real, shim)}."""
    out = {}
    for case in CASES:
        wd = tmp_path_factory.mktemp(case)
        shim = _run(sys.executable, case, str(wd / "shim.json"), str(wd))
        real = None
        if HAS_REAL:
            real = _run(PV_REAL, case, str(wd / "real.json"), str(wd))
        out[case] = (real, shim)
    return out


def _diff(a, b, path, tol, ignores, problems):
    key = path[1:] if path.startswith(".") else path
    if any(key == ig or key.startswith(ig + ".") or key.startswith(ig + "[") for ig in ignores):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                problems.append(f"{path}.{k}: only in shim ({b[k]!r})")
            elif k not in b:
                problems.append(f"{path}.{k}: only in real ({a[k]!r})")
            else:
                _diff(a[k], b[k], f"{path}.{k}", tol, ignores, problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: len {len(a)} (real) vs {len(b)} (shim): {a!r} vs {b!r}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", tol, ignores, problems)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        t = 1e-3
        best = -1
        for k, v in tol.items():
            if (key == k or key.startswith(k + ".") or key.startswith(k + "[")) and len(k) > best:
                best = len(k)
                t = v
        if abs(a - b) > max(t * max(abs(a), abs(b)), 1e-6):
            problems.append(f"{path}: real={a} shim={b} (tol {t})")
    elif a != b:
        problems.append(f"{path}: real={a!r} shim={b!r}")


def _case_config(case):
    import importlib.util

    spec = importlib.util.spec_from_file_location("case_mod", os.path.join(CASES_DIR, case + ".py"))
    mod = importlib.util.module_from_spec(spec)
    # do not execute main(); just read module-level IGNORE/TOL
    src = open(spec.origin).read().replace("main(", "_unused_main(")
    ns = {"__file__": spec.origin}
    exec(compile(src, spec.origin, "exec"), ns)
    return ns.get("IGNORE", []), ns.get("TOL", {})


@pytest.mark.skipif(not HAS_REAL, reason=f"real ParaView python not found at {PV_REAL}")
@pytest.mark.parametrize("case", CASES)
def test_differential(traces, case):
    real, shim = traces[case]
    if real is None:
        pytest.skip("no real trace")
    ignores, tol = _case_config(case)
    ignores = list(ignores) + ["misc.out_dir", "render.size"]
    # only compare render success, not pixel content
    if isinstance(real.get("render"), dict) and isinstance(shim.get("render"), dict):
        real["render"] = {"ok": real["render"].get("ok")}
        shim["render"] = {"ok": shim["render"].get("ok")}
    problems = []
    _diff(real, shim, "", tol, ignores, problems)
    assert not problems, f"case {case} differs:\n" + "\n".join(problems[:40])


@pytest.mark.parametrize("case", CASES)
def test_case_runs_under_shim(traces, case):
    """Every differential case must at least run cleanly under the shim."""
    _real, shim = traces[case]
    assert shim.get("fetches"), f"case {case} produced no fetch traces"
