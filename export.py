"""Export the best checkpoint to Hugging Face format and a downloadable GGUF.

Colab cells (run after model.py finishes):
    %pip install -q -r requirements-export.txt
    %run export.py --setup --download

Optional Hub upload, using notebook_login() or a Colab secret named HF_TOKEN:
    %run export.py --setup --repo-id YOUR_NAME/christmasmodel --download
New Hub repos are private unless --public is supplied. No upload without repo-id.
Use %run (not !python) for the browser download, which needs the notebook kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LLAMA_COMMIT = "7d7f0287530bc8981690f361d8c1627dbae1c99a"
TOKENIZER_PROBES = [
    "On Christmas Eve, the snow fell softly.",
    "I'm here; she'll come. WE'RE ready! 12345 3.14",
    "  spaces\tand\n\nnew lines\r\nend ",
    "“Merry Christmas!” — café, naïve, £5. 雪 🎄",
    "<|endoftext|>A story<|pad|>",
]


def run(command, **kwargs):
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def ensure_converter(path, setup):
    path = Path(path).resolve()
    if not path.exists():
        if not setup:
            raise ValueError("llama.cpp missing; pass --setup to fetch the pinned converter")
        run(["git", "clone", "--filter=blob:none", "--depth", "1", "--no-checkout",
             "https://github.com/ggml-org/llama.cpp.git", path])
        run(["git", "-C", path, "fetch", "--depth", "1", "origin", LLAMA_COMMIT])
        run(["git", "-C", path, "checkout", "--detach", LLAMA_COMMIT])
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != LLAMA_COMMIT:
        raise ValueError(f"Use a separate llama.cpp checkout at {LLAMA_COMMIT}; found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("Pinned llama.cpp has tracked modifications; use a clean checkout")
    return path


def validate_tokenizer(tokenizer):
    """Custom BPE merges change llama.cpp's hash even with GPT-2 pre-tokenization.

    Only accept the exact unnormalized byte-level pipeline trained by model.py.
    Do not silently label an arbitrary custom tokenizer as GPT-2 compatible.
    """
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    pre = backend["pre_tokenizer"]
    if (backend["model"]["type"] != "BPE" or backend["normalizer"] is not None
            or pre != {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
            or backend["decoder"]["type"] != "ByteLevel"
            or backend["post_processor"]["type"] != "ByteLevel"):
        raise ValueError("GGUF export supports only this project's GPT-2 byte-level BPE tokenizer")
    if backend["model"].get("dropout") or backend["model"].get("byte_fallback"):
        raise ValueError("Unsupported BPE dropout/byte fallback")
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.add_prefix_space:
        raise ValueError("Expected distinct pad/EOS tokens and no implicit prefix space")
    for probe in TOKENIZER_PROBES:
        ids = tokenizer.encode(probe, add_special_tokens=False)
        if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != probe:
            raise ValueError(f"Tokenizer round trip failed for {probe!r}")
        if tokenizer.encode(probe, add_special_tokens=True) != ids:
            raise ValueError("Expected no automatic BOS/EOS insertion")


def convert_gguf(hf_dir, outfile, llama_dir, outtype):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(llama_dir))
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf
    from conversion.gpt2 import GPT2Model
    validate_tokenizer(AutoTokenizer.from_pretrained(hf_dir, local_files_only=True))

    class ChristmasGPT2(GPT2Model):
        model_arch = gguf.MODEL_ARCH.GPT2

        def get_vocab_base_pre(self, tokenizer):
            validate_tokenizer(tokenizer)
            return "gpt-2"

    types = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
             "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}
    logging.basicConfig(level=logging.INFO)
    with torch.inference_mode():
        converter = ChristmasGPT2(hf_dir, types[outtype], outfile, eager=True)
        converter.write()
    with outfile.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError("Converter did not create a GGUF file")


def verify_gguf(hf_dir, gguf_path, llama_dir):
    """Compile a CPU server; compare token IDs and greedy first tokens to HF."""
    import requests
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel
    binary = llama_dir / "build" / "bin" / "llama-server"
    if not binary.exists():
        run(["cmake", "-S", llama_dir, "-B", llama_dir / "build", "-DCMAKE_BUILD_TYPE=Release",
             "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DLLAMA_OPENSSL=OFF",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
             "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF"])
        run(["cmake", "--build", llama_dir / "build", "--target", "llama-server", "-j", "2"])
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(hf_dir, local_files_only=True).eval()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = gguf_path.with_suffix(".verify.log")
    with log_path.open("w") as log:
        process = subprocess.Popen([str(binary), "-m", str(gguf_path), "--host", "127.0.0.1",
                                    "--port", str(port), "-c", str(model.config.n_positions),
                                    "-ngl", "0", "--parallel", "1"], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited; inspect {log_path}")
                try:
                    if requests.get(base + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError(f"llama-server startup timed out; inspect {log_path}")
            for probe in TOKENIZER_PROBES:
                response = requests.post(base + "/tokenize", json={"content": probe,
                                         "add_special": False, "parse_special": True}, timeout=30)
                response.raise_for_status()
                actual = response.json()["tokens"]
                expected = tokenizer.encode(probe, add_special_tokens=False)
                if actual != expected:
                    raise RuntimeError(f"GGUF tokenizer mismatch on {probe!r}: {actual} != {expected}")
            matches = []
            for probe in TOKENIZER_PROBES[:3]:
                inputs = tokenizer(probe, return_tensors="pt")
                with torch.inference_mode():
                    expected = int(model(**inputs).logits[0, -1].argmax())
                response = requests.post(base + "/completion", json={
                    "prompt": inputs["input_ids"][0].tolist(), "n_predict": 1, "temperature": 0,
                    "repeat_penalty": 1., "cache_prompt": False, "return_tokens": True,
                }, timeout=60)
                response.raise_for_status()
                actual = response.json().get("tokens")
                if actual is None:
                    raise RuntimeError("llama-server omitted token IDs; verification unavailable")
                matches.append({"prompt": probe, "hf_token": expected, "gguf_tokens": actual,
                                "match": actual == [expected]})
            if not all(row["match"] for row in matches):
                raise RuntimeError(f"First-token mismatch (rounding/quantization can change near ties): {matches}")
            return {"tokenizer_probes_passed": len(TOKENIZER_PROBES), "first_token_checks": matches}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def hub_api():
    from huggingface_hub import HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception as exc:
            raise RuntimeError("Log in with huggingface_hub.notebook_login() or set the HF_TOKEN Colab secret") from exc
    api = HfApi(token=token)
    api.whoami()  # Fail before conversion if the supplied token is invalid.
    return api


def upload_artifacts(api, args, hf_dir, gguf_path):
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(hf_dir),
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "README.md"],
                      commit_message="Upload Christmasmodel weights and tokenizer")
    api.upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=str(gguf_path),
                    path_in_repo=gguf_path.name, commit_message="Add GGUF weights")
    api.upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=str(gguf_path.parent / "export_manifest.json"),
                    path_in_repo="export_manifest.json", commit_message="Record GGUF conversion provenance")
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")


def download(gguf_path):
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("--download needs Colab; elsewhere use the printed local artifact paths") from exc
    files.download(str(gguf_path))


def export_model(args):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    source, out = Path(args.model_dir).resolve(), Path(args.output_dir).resolve()
    api = hub_api() if args.repo_id else None
    if args.upload_only:
        if api is None:
            raise ValueError("--upload-only requires --repo-id")
        manifest = json.loads((out / "export_manifest.json").read_text())
        if manifest["outtype"] not in ("f16", "f32", "q8_0"):
            raise ValueError("Unknown export format in manifest")
        gguf_path = out / f"christmasmodel-{manifest['outtype'].upper()}.gguf"
        if hashlib.sha256(gguf_path.read_bytes()).hexdigest() != manifest["gguf_sha256"]:
            raise ValueError("GGUF differs from its export manifest")
        upload_artifacts(api, args, out / "huggingface", gguf_path)
        if args.download:
            download(gguf_path)
        return gguf_path
    if not (source / "config.json").exists():
        raise ValueError(f"No trained Hugging Face checkpoint at {source}; run model.py train first")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export directory is not empty; choose a new --output-dir")
    llama_dir = ensure_converter(args.llama_dir, args.setup)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(source, local_files_only=True)
    validate_tokenizer(tokenizer)
    cfg = model.config
    if (cfg.n_inner not in (None, 4*cfg.n_embd) or cfg.activation_function != "gelu_new"
            or cfg.scale_attn_by_inverse_layer_idx or not cfg.scale_attn_weights
            or cfg.add_cross_attention or cfg.n_embd % cfg.n_head
            or cfg.vocab_size != len(tokenizer)):
        raise ValueError("Model differs from the GPT-2 architecture supported by this exporter")
    if (cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id) != (
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        raise ValueError("Model and tokenizer special token IDs disagree")
    cfg.n_ctx = cfg.n_positions  # Required by the pinned GPT-2 converter.
    cfg.use_cache = True
    hf_dir = out / "huggingface"
    hf_dir.mkdir(parents=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    tokenizer.save_pretrained(hf_dir)
    for name in ("README.md", "training_summary.json", "samples.json", "supplementary_sources.json"):
        if (source / name).exists():
            shutil.copy2(source / name, hf_dir / name)
    del model
    gguf_path = out / f"christmasmodel-{args.outtype.upper()}.gguf"
    convert_gguf(hf_dir, gguf_path, llama_dir, args.outtype)
    verification = verify_gguf(hf_dir, gguf_path, llama_dir) if args.verify else {"runtime_verified": False}
    manifest = {"llama_cpp_commit": LLAMA_COMMIT, "outtype": args.outtype,
                "gguf_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
                "bytes": gguf_path.stat().st_size, "verification": verification}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = shutil.make_archive(str(out / "christmasmodel-huggingface"), "zip", hf_dir)
    if args.repo_id:
        upload_artifacts(api, args, hf_dir, gguf_path)
    print(f"HF directory: {hf_dir}\nHF ZIP: {archive}\nGGUF: {gguf_path}")
    if args.download:
        download(gguf_path)
    return gguf_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(ROOT / "runs" / "christmas" / "best"))
    parser.add_argument("--output-dir", default=str(ROOT / "exports" / "christmas"))
    parser.add_argument("--llama-dir", default=str(ROOT / "llama.cpp"))
    parser.add_argument("--setup", action="store_true", help="Fetch the pinned llama.cpp source if missing")
    parser.add_argument("--outtype", choices=["f16", "f32", "q8_0"], default="f16")
    parser.add_argument("--verify", action="store_true", help="Build CPU llama-server and verify tokenizer/first tokens")
    parser.add_argument("--repo-id", help="Optional Hugging Face upload destination: owner/model-name")
    parser.add_argument("--upload-only", action="store_true", help="Upload/retry an existing export without converting again")
    parser.add_argument("--public", action="store_true", help="Create a public Hub repo (default: private)")
    parser.add_argument("--download", action="store_true", help="Colab browser download; invoke with %%run")
    args = parser.parse_args(argv)
    export_model(args)


if __name__ == "__main__":
    main()
