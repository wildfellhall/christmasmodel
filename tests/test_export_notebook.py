"""Exercise the actual export entry point with Colab's injected kernel argv."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as exporter


def load_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = Path(exporter.__file__).read_text()
    namespace = {"__name__": "loaded_cell"}
    exec(compile(source, "<export-cell>", "exec"), namespace)
    # Retain all real definitions; stub only installation and artifact creation.
    installs, exports = [], []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: installs.append((command, kwargs)))
    namespace["export_model"] = lambda args: exports.append(vars(args))
    namespace["__name__"] = "__main__"
    entrypoint = compile(ast.Module(body=[ast.parse(source).body[-1]], type_ignores=[]), "<entrypoint>", "exec")
    return namespace, entrypoint, installs, exports


def test_pasted_export_ignores_colab_kernel_flags(monkeypatch, tmp_path):
    namespace, entrypoint, installs, exports = load_cell(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["colab_kernel_launcher.py", "-f", "/runtime/kernel.json"])
    exec(entrypoint, namespace)
    assert namespace["ROOT"] == tmp_path
    assert installs[0][0][1:5] == ["-m", "pip", "install", "-q"]
    assert installs[0][1]["check"] is True
    assert exports[0]["model_dir"] == "runs/story-scratch"
    assert exports[0]["setup"] and exports[0]["verify"] and exports[0]["download"]
    assert exports[0]["repo_id"] is None


def test_pasted_export_honors_edited_notebook_paths(monkeypatch, tmp_path):
    namespace, entrypoint, _, exports = load_cell(monkeypatch, tmp_path)
    namespace["EXPORT_NOTEBOOK_ARGS"] = ["--model-dir", "/content/drive/MyDrive/model/best",
                                          "--output-dir", "/content/exports/reviewed", "--outtype", "q8_0"]
    monkeypatch.setattr(sys, "argv", ["colab_kernel_launcher.py", "-f", "kernel.json"])
    exec(entrypoint, namespace)
    assert exports[0]["model_dir"] == "/content/drive/MyDrive/model/best"
    assert exports[0]["output_dir"] == "/content/exports/reviewed"
    assert exports[0]["outtype"] == "q8_0"


def test_script_entrypoint_keeps_explicit_cli_arguments(monkeypatch, tmp_path):
    namespace, entrypoint, installs, exports = load_cell(monkeypatch, tmp_path)
    namespace["__file__"] = exporter.__file__
    monkeypatch.setattr(sys, "argv", ["export.py", "--model-dir", "chosen/best", "--outtype", "f32"])
    exec(entrypoint, namespace)
    assert not installs
    assert exports[0]["model_dir"] == "chosen/best" and exports[0]["outtype"] == "f32"
    monkeypatch.setattr(sys, "argv", ["export.py", "--misspelled-option"])
    with pytest.raises(SystemExit) as error:
        exec(entrypoint, namespace)
    assert error.value.code == 2


def test_notebook_options_remain_strict(monkeypatch, tmp_path):
    namespace, entrypoint, _, exports = load_cell(monkeypatch, tmp_path)
    namespace["EXPORT_NOTEBOOK_ARGS"] = ["--modle-dir", "typo/best"]
    with pytest.raises(SystemExit) as error:
        exec(entrypoint, namespace)
    assert error.value.code == 2 and not exports
