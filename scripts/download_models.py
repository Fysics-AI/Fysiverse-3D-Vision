#!/usr/bin/env python3
"""Download Fysiverse-3D-Vision model artifacts.

Model files are intentionally kept outside Git. Repository names, official
URLs, local paths, and required files are defined in configs/models.json.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import socket
import ssl
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_SOURCE_CONFIG = PROJECT_ROOT / "configs" / "sources.json"
MODEL_SOURCES = ("auto", "upstream", "modelscope")


class ModelScopeAccessError(RuntimeError):
    """ModelScope rejected access to a configured model repository."""

    def __init__(self, repo_id: str, original: BaseException) -> None:
        self.repo_id = repo_id
        self.original = original
        super().__init__(f"ModelScope access failed for {repo_id}: {original}")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_names(requested_model: str, models: dict[str, Any]) -> list[str]:
    roots = list(models) if requested_model == "all" else [requested_model]
    unknown = [name for name in roots if name not in models]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(models)}"
        )
    selected: list[str] = []
    visiting: set[str] = set()

    def add(name: str) -> None:
        if name in selected:
            return
        if name in visiting:
            raise ValueError(f"Cyclic model dependency involving: {name}")
        if name not in models:
            raise ValueError(f"Unknown model dependency: {name}")
        visiting.add(name)
        selected.append(name)
        for dependency in models[name].get("model_dependencies", []):
            add(str(dependency))
        visiting.remove(name)

    for root in roots:
        add(root)
    return selected


def fetch_sources(
    requested_model: str,
    *,
    project_root: Path,
    model_config: Path,
    source_config: Path,
    check_only: bool,
    network_timeout: float,
    source_transfer_timeout: float | None,
) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "fetch_sources.py"),
        "--model",
        requested_model,
        "--project-root",
        str(project_root),
        "--model-config",
        str(model_config),
        "--source-config",
        str(source_config),
        "--network-timeout",
        str(network_timeout),
    ]
    if source_transfer_timeout is not None:
        command.extend(["--transfer-timeout", str(source_transfer_timeout)])
    if check_only:
        command.append("--check-only")
    result = subprocess.run(command, check=False)
    if result.returncode:
        action = "validation" if check_only else "download"
        raise RuntimeError(f"required official source {action} failed")


def check_model(name: str, spec: dict[str, Any], project_root: Path) -> list[str]:
    model_dir = project_root / spec["local_dir"]
    if not model_dir.is_dir():
        return [f"{model_dir} (directory does not exist)"]

    payload_entries = [
        path for path in model_dir.iterdir() if path.name not in {".cache", ".huggingface"}
    ]
    missing = [] if payload_entries else [f"{model_dir} (directory has no model files)"]
    for relative_path in spec.get("required_files", []):
        if not (model_dir / relative_path).is_file():
            missing.append(str(model_dir / relative_path))
    for relative_path, expected_size in spec.get("required_file_sizes", {}).items():
        path = model_dir / relative_path
        if not path.is_file():
            if str(path) not in missing:
                missing.append(str(path))
        elif path.stat().st_size != int(expected_size):
            missing.append(
                f"{path} (size {path.stat().st_size}, expected {int(expected_size)} bytes)"
            )
    for entry in spec.get("files") or []:
        path = model_dir / str(entry["path"])
        if not path.is_file():
            if str(path) not in missing:
                missing.append(str(path))
            continue
        missing.extend(_file_integrity_issues(path, entry))
    for pattern in spec.get("required_globs", []):
        if not any(path.is_file() for path in model_dir.glob(pattern)):
            missing.append(f"{model_dir / pattern} (no matching files)")
    return missing


def _file_integrity_issues(path: Path, metadata: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_size = metadata.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        issues.append(
            f"{path} (size {path.stat().st_size}, expected {int(expected_size)} bytes)"
        )
        return issues
    expected_sha256 = str(metadata.get("sha256") or "").lower()
    if expected_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            issues.append(
                f"{path} (SHA-256 {actual_sha256}, expected {expected_sha256})"
            )
    return issues


def validate_model_package(name: str, spec: dict[str, Any], project_root: Path) -> None:
    if name != "layout":
        return
    model_dir = project_root / spec["local_dir"]
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_layout_model.py"),
            "--model-dir",
            str(model_dir),
        ],
        check=False,
    )
    if result.returncode:
        raise RuntimeError("layout package content validation failed")


def check_disk_space(
    names: list[str],
    models: dict[str, Any],
    project_root: Path,
    config: dict[str, Any],
    *,
    skip: bool,
) -> None:
    estimated_gb = sum(float(models[name].get("estimated_size_gb", 0.0)) for name in names)
    safety_factor = float(config.get("download_safety_factor", 1.25))
    required_gb = estimated_gb * safety_factor
    is_all_models = set(names) == set(models)
    if is_all_models:
        required_gb = max(required_gb, float(config.get("minimum_free_gb_for_all_models", required_gb)))
    free_gb = shutil.disk_usage(project_root).free / (1024**3)
    print(
        f"[disk] free={free_gb:.1f} GiB, estimated download={estimated_gb:.1f} GiB, "
        f"required with safety margin={required_gb:.1f} GiB"
    )
    if skip:
        print("[disk] check skipped by explicit --skip-disk-check", file=sys.stderr)
        return
    if free_gb < required_gb:
        selection = "all" if is_all_models else names[0]
        raise SystemExit(
            f"Download not started: insufficient disk space. {free_gb:.1f} GiB available, "
            f"at least {required_gb:.1f} GiB is required for the selected models. "
            "Existing files were not deleted. Free space or select fewer models, then retry.\n"
            f"Retry: python scripts/download_models.py --model {selection}"
        )


def _set_response_socket_timeout(response: Any, timeout_seconds: float) -> None:
    """Best-effort reduction of the active socket timeout to the route deadline."""
    candidates = [response, getattr(response, "fp", None)]
    fp = getattr(response, "fp", None)
    candidates.extend([getattr(fp, "raw", None), getattr(getattr(fp, "raw", None), "_sock", None)])
    for candidate in candidates:
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            setter(max(timeout_seconds, 0.001))
            return


def _download_url(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Download and validate one official file before publishing it atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    deadline = time.monotonic() + timeout_seconds
    digest = hashlib.sha256()
    byte_count = 0
    try:
        request = Request(url, headers={"User-Agent": "Fysiverse-3D-Vision-model-downloader/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as output:
            reader = getattr(response, "read1", None) or response.read
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"route exceeded the {timeout_seconds:g}-second wall-clock limit"
                    )
                _set_response_socket_timeout(response, remaining)
                chunk = reader(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"route exceeded the {timeout_seconds:g}-second wall-clock limit"
                    )
        if expected_size is not None and byte_count != int(expected_size):
            raise RuntimeError(
                f"downloaded size {byte_count}, expected {int(expected_size)} bytes"
            )
        if expected_sha256 and digest.hexdigest() != expected_sha256.lower():
            raise RuntimeError(
                f"downloaded SHA-256 {digest.hexdigest()}, expected {expected_sha256.lower()}"
            )
        os.replace(temporary, destination)
    except (HTTPError, URLError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"official URL download failed: {url}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _download_huggingface(
    name: str,
    spec: dict[str, Any],
    *,
    local_dir: Path,
    revision: str | None,
    token: str | None,
    timeout_seconds: float,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: "
            "python -m pip install -U huggingface_hub"
        ) from exc

    source_revision = revision or spec.get("revision")
    print(
        f"[{name}] {spec['source_type']} {spec['repo_id']} -> {local_dir}"
        + (f" @ {source_revision}" if source_revision else ""),
        flush=True,
    )
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=source_revision,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=spec.get("allow_patterns") or None,
        etag_timeout=timeout_seconds,
    )


def _download_official_files(
    name: str,
    spec: dict[str, Any],
    local_dir: Path,
    *,
    timeout_seconds: float,
) -> None:
    files = spec.get("files") or []
    if not files:
        raise ValueError(f"{name} uses official_url but has no files in configs/models.json")
    print(f"[{name}] official upstream files -> {local_dir}", flush=True)
    for entry in files:
        url = str(entry["url"])
        relative_path = Path(str(entry["path"]))
        destination = local_dir / relative_path
        if destination.is_file() and not _file_integrity_issues(destination, entry):
            print(f"[{name}] keep existing {relative_path}", flush=True)
            continue
        if destination.is_file():
            print(
                f"[{name}] existing {relative_path} failed integrity validation; downloading a replacement",
                file=sys.stderr,
                flush=True,
            )
        download_urls = [str(value) for value in entry.get("download_urls") or [url]]
        if url not in download_urls:
            download_urls.append(url)
        errors: list[str] = []
        for index, download_url in enumerate(download_urls, start=1):
            print(
                f"[{name}] {download_url} -> {relative_path} "
                f"(route {index}/{len(download_urls)})",
                flush=True,
            )
            try:
                _download_url(
                    download_url,
                    destination,
                    timeout_seconds=timeout_seconds,
                    expected_size=(
                        int(entry["size_bytes"]) if entry.get("size_bytes") is not None else None
                    ),
                    expected_sha256=str(entry.get("sha256") or "") or None,
                )
                break
            except RuntimeError as exc:
                errors.append(str(exc))
                if index < len(download_urls):
                    print(f"[{name}] route failed; trying the next configured URL", file=sys.stderr)
        else:
            raise RuntimeError("all configured download URLs failed: " + "; ".join(errors))


def _matches_patterns(relative_path: Path, patterns: list[str]) -> bool:
    value = relative_path.as_posix()
    return not patterns or any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def _install_modelscope_snapshot(
    snapshot_root: Path,
    local_dir: Path,
    *,
    allow_patterns: list[str],
) -> int:
    installed = 0
    for source in snapshot_root.rglob("*"):
        relative_path = source.relative_to(snapshot_root)
        if not source.is_file() or any(part in {".git", ".cache"} for part in relative_path.parts):
            continue
        if not _matches_patterns(relative_path, allow_patterns):
            continue
        destination = local_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".modelscope.part", dir=str(destination.parent)
        )
        os.close(fd)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            try:
                os.link(source.resolve(strict=True), temporary)
            except OSError:
                shutil.copy2(source.resolve(strict=True), temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        installed += 1
    return installed


def _download_modelscope(
    name: str,
    spec: dict[str, Any],
    *,
    repo_id: str,
    local_dir: Path,
    token: str | None,
) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope fallback requires the modelscope SDK. Install it with: "
            "python -m pip install 'modelscope>=1.22,<2.0'"
        ) from exc

    revision = spec.get("modelscope_revision")
    allow_patterns = [str(pattern) for pattern in spec.get("allow_patterns") or []]
    print(
        f"[{name}] modelscope {repo_id} -> {local_dir}"
        + (f" @ {revision}" if revision else ""),
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{name}.modelscope.", dir=str(local_dir.parent)
    ) as cache_dir:
        kwargs: dict[str, Any] = {
            "model_id": repo_id,
            "cache_dir": cache_dir,
        }
        if token:
            # Pass the token only for this request; HubApi.login() persists it.
            kwargs["token"] = token
        if revision:
            kwargs["revision"] = str(revision)
        if allow_patterns:
            kwargs["allow_file_pattern"] = allow_patterns
        snapshot_root = Path(snapshot_download(**kwargs)).resolve()
        if not snapshot_root.is_dir():
            raise RuntimeError(f"ModelScope returned an invalid snapshot path: {snapshot_root}")
        installed = _install_modelscope_snapshot(
            snapshot_root,
            local_dir,
            allow_patterns=allow_patterns,
        )
    if installed == 0:
        raise RuntimeError(f"ModelScope snapshot for {repo_id} contained no selected model files")


def _download_upstream(
    name: str,
    spec: dict[str, Any],
    *,
    local_dir: Path,
    revision: str | None,
    token: str | None,
    timeout_seconds: float,
) -> None:
    source_type = spec.get("source_type", "company_hf")
    if source_type in {"company_hf", "official_hf"}:
        _download_huggingface(
            name,
            spec,
            local_dir=local_dir,
            revision=revision,
            token=token,
            timeout_seconds=timeout_seconds,
        )
    elif source_type == "official_url":
        _download_official_files(
            name,
            spec,
            local_dir,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported source_type for {name}: {source_type}")


def parse_modelscope_repo_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, repo_id = value.partition("=")
        if not separator or not name.strip() or not repo_id.strip():
            raise ValueError(
                f"Invalid --modelscope-repo value: {value!r}; expected MODEL=OWNER/REPOSITORY"
            )
        overrides[name.strip()] = repo_id.strip()
    return overrides


def resolve_modelscope_repo(
    name: str,
    spec: dict[str, Any],
    overrides: dict[str, str],
) -> str | None:
    if name in overrides:
        return overrides[name]
    environment_key = "FYSIVERSE_MODELSCOPE_REPO_" + "".join(
        character if character.isalnum() else "_" for character in name.upper()
    )
    return os.environ.get(environment_key) or spec.get("modelscope_repo_id")


def download_model(
    name: str,
    spec: dict[str, Any],
    *,
    project_root: Path,
    revision: str | None,
    token: str | None,
    modelscope_token: str | None,
    model_source: str,
    modelscope_repo_overrides: dict[str, str],
    timeout_seconds: float,
) -> None:
    local_dir = project_root / spec["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    modelscope_repo = resolve_modelscope_repo(name, spec, modelscope_repo_overrides)
    if model_source == "modelscope":
        if not modelscope_repo:
            raise RuntimeError(
                f"{name} has no configured ModelScope repository. Add modelscope_repo_id "
                "to configs/models.json or pass "
                f"--modelscope-repo {name}=OWNER/REPOSITORY."
            )
        try:
            _download_modelscope(
                name,
                spec,
                repo_id=modelscope_repo,
                local_dir=local_dir,
                token=modelscope_token,
            )
        except Exception as modelscope_error:
            if is_hf_access_error(modelscope_error):
                raise ModelScopeAccessError(modelscope_repo, modelscope_error) from modelscope_error
            raise
    else:
        try:
            _download_upstream(
                name,
                spec,
                local_dir=local_dir,
                revision=revision,
                token=token,
                timeout_seconds=timeout_seconds,
            )
        except Exception as upstream_error:
            if model_source != "auto" or is_hf_access_error(upstream_error):
                raise
            if not is_network_error(upstream_error):
                raise
            if not modelscope_repo:
                raise RuntimeError(
                    f"{name} upstream network download failed and no ModelScope fallback is "
                    "configured. Add modelscope_repo_id to configs/models.json or pass "
                    f"--modelscope-repo {name}=OWNER/REPOSITORY. "
                    f"Upstream error: {upstream_error}"
                ) from upstream_error
            print(
                f"[{name}] upstream network failed; trying ModelScope {modelscope_repo}",
                file=sys.stderr,
                flush=True,
            )
            try:
                _download_modelscope(
                    name,
                    spec,
                    repo_id=modelscope_repo,
                    local_dir=local_dir,
                    token=modelscope_token,
                )
            except Exception as modelscope_error:
                if is_hf_access_error(modelscope_error):
                    raise ModelScopeAccessError(modelscope_repo, modelscope_error) from modelscope_error
                raise RuntimeError(
                    f"{name} upstream network download and ModelScope fallback both failed. "
                    f"Upstream error: {upstream_error}. "
                    f"ModelScope error: {modelscope_error}"
                ) from modelscope_error

    missing = check_model(name, spec, project_root)
    if missing:
        details = "\n".join(f"  {path}" for path in missing)
        raise RuntimeError(f"download finished but model validation failed:\n{details}")
    validate_model_package(name, spec, project_root)
    print(f"[{name}] ready", flush=True)


def is_hf_access_error(exc: BaseException) -> bool:
    """Return whether an exception commonly means authentication/access is needed."""
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return True

    class_name = type(exc).__name__.lower()
    if class_name in {"gatedrepoerror", "huggingfacehubhttperror"}:
        message = str(exc).lower()
        return any(token in message for token in ("gated", "unauthorized", "forbidden", "permission", "token"))

    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "401 client error",
            "403 client error",
            "gated repo",
            "accept the user conditions",
            "access request",
            "unauthorized",
            "forbidden",
            "invalid token",
        )
    )


def is_network_error(exc: BaseException) -> bool:
    """Return whether an exception represents a retryable external network failure."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if isinstance(current, HTTPError):
            status_code = current.code
        if status_code is not None:
            try:
                numeric_status = int(status_code)
            except (TypeError, ValueError):
                numeric_status = 0
            if numeric_status in {408, 425, 429, 500, 502, 503, 504}:
                return True
            if numeric_status >= 400:
                current = current.__cause__ or current.__context__
                continue
        if isinstance(current, (TimeoutError, socket.timeout, ssl.SSLError)):
            return True
        if isinstance(current, URLError) and not isinstance(current, HTTPError):
            return True
        class_name = type(current).__name__.lower()
        if any(token in class_name for token in ("timeout", "connectionerror", "proxyerror", "sslerror")):
            return True
        message = str(current).lower()
        if any(
            token in message
            for token in (
                "timed out",
                "timeout",
                "could not resolve host",
                "name or service not known",
                "temporary failure in name resolution",
                "connection reset",
                "connection aborted",
                "connection refused",
                "remote disconnected",
                "tls handshake",
                "ssl certificate",
                "network is unreachable",
                "proxy error",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def print_hf_access_help(
    name: str,
    spec: dict[str, Any],
    exc: BaseException,
    requested_model: str,
) -> None:
    repo_id = spec.get("repo_id", "the configured Hugging Face repository")
    print(
        f"[{name}] Hugging Face access is required for {repo_id}; download stopped.",
        file=sys.stderr,
    )
    print("Use one of these supported authentication methods, then run the same command again:", file=sys.stderr)
    print("  huggingface-cli login", file=sys.stderr)
    print("  # or: hf auth login", file=sys.stderr)
    print("  export HF_TOKEN=hf_...", file=sys.stderr)
    print(
        f"If this is a gated model, open {spec.get('source_url', f'https://huggingface.co/{repo_id}')}",
        file=sys.stderr,
    )
    print("and accept the model terms with the same account before retrying.", file=sys.stderr)
    print("The downloader never asks for or stores your Hugging Face password.", file=sys.stderr)
    print(f"Retry: python scripts/download_models.py --model {requested_model}", file=sys.stderr)
    print(f"Original error: {exc}", file=sys.stderr)


def print_modelscope_access_help(
    exc: ModelScopeAccessError,
    requested_model: str,
    model_source: str,
) -> None:
    print(
        f"ModelScope access is required for {exc.repo_id}; download stopped.",
        file=sys.stderr,
    )
    print(
        "Create an API token in the ModelScope account settings, confirm that the "
        "account can access the repository, then retry:",
        file=sys.stderr,
    )
    print("  export MODELSCOPE_API_TOKEN=...", file=sys.stderr)
    print(
        "  # or pass: --modelscope-token \"$MODELSCOPE_API_TOKEN\"",
        file=sys.stderr,
    )
    print("The downloader never asks for a ModelScope password.", file=sys.stderr)
    print(
        f"Retry: python scripts/download_models.py --model {requested_model} "
        f"--model-source {model_source}",
        file=sys.stderr,
    )
    print(f"Original error: {exc.original}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        help="Model key from configs/models.json, or all (default: all).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision for every model.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--model-source",
        choices=MODEL_SOURCES,
        default=os.environ.get("FYSIVERSE_MODEL_SOURCE", "auto"),
        help=(
            "Model artifact provider: auto tries configured upstream first and falls back "
            "to ModelScope only on network errors (default: auto)."
        ),
    )
    parser.add_argument(
        "--modelscope-token",
        default=os.environ.get("MODELSCOPE_API_TOKEN"),
        help="Optional ModelScope API token; defaults to MODELSCOPE_API_TOKEN.",
    )
    parser.add_argument(
        "--modelscope-repo",
        action="append",
        default=[],
        metavar="MODEL=OWNER/REPOSITORY",
        help="Override/add a ModelScope repository for one model; repeat as needed.",
    )
    parser.add_argument(
        "--network-timeout",
        type=float,
        default=None,
        help=(
            "Direct-model-file route limit, Hugging Face Hub metadata timeout, "
            "and Git source route-probe/continuous-low-speed threshold."
        ),
    )
    parser.add_argument(
        "--skip-source-fetch",
        action="store_true",
        help="Skip pinned Git source fetch/validation when sources were prepared separately.",
    )
    parser.add_argument(
        "--source-transfer-timeout",
        type=float,
        default=None,
        help=(
            "Optional wall-clock limit for each Git fetch/submodule attempt; "
            "defaults to configs/sources.json."
        ),
    )
    parser.add_argument("--check-only", action="store_true", help="Only check required local files.")
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the conservative disk-space check (not recommended).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    models = config["models"]
    try:
        names = resolve_model_names(args.model, models)
        modelscope_repo_overrides = parse_modelscope_repo_overrides(args.modelscope_repo)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    timeout_seconds = float(
        args.network_timeout
        if args.network_timeout is not None
        else config.get("network_timeout_seconds", 30)
    )
    if timeout_seconds <= 0:
        raise SystemExit("--network-timeout must be positive")
    if args.source_transfer_timeout is not None and args.source_transfer_timeout <= 0:
        raise SystemExit("--source-transfer-timeout must be positive")
    project_root = args.project_root.resolve()
    if args.model_source == "modelscope" and not args.check_only:
        missing_modelscope = [
            name
            for name in names
            if check_model(name, models[name], project_root)
            if not resolve_modelscope_repo(name, models[name], modelscope_repo_overrides)
        ]
        if missing_modelscope:
            raise SystemExit(
                "ModelScope-only download cannot start because these models have no "
                f"configured repository: {', '.join(missing_modelscope)}. "
                "Add modelscope_repo_id entries or pass --modelscope-repo MODEL=OWNER/REPOSITORY."
            )
    if not args.check_only:
        check_disk_space(
            names,
            models,
            project_root,
            config,
            skip=args.skip_disk_check,
        )
    if args.skip_source_fetch:
        print("[source] fetch/validation skipped by explicit --skip-source-fetch", file=sys.stderr)
    else:
        try:
            fetch_sources(
                args.model,
                project_root=project_root,
                model_config=args.config.resolve(),
                source_config=args.source_config.resolve(),
                check_only=args.check_only,
                network_timeout=timeout_seconds,
                source_transfer_timeout=args.source_transfer_timeout,
            )
        except RuntimeError as exc:
            print(f"[source] {exc}", file=sys.stderr)
            print(
                "[source] ModelScope handles model artifacts only. Pinned Git sources probe "
                "github.com, ghproxy.net, and ghfast.top, then prefer the last healthy route. "
                "Check connectivity or "
                "prepare sources first and retry with --skip-source-fetch.",
                file=sys.stderr,
            )
            return 1
    failed = False
    for name in names:
        spec = models[name]
        if args.check_only:
            missing = check_model(name, spec, project_root)
            if missing:
                failed = True
                print(f"[{name}] missing required files:", file=sys.stderr)
                for path in missing:
                    print(f"  {path}", file=sys.stderr)
            else:
                try:
                    validate_model_package(name, spec, project_root)
                except RuntimeError as exc:
                    failed = True
                    print(f"[{name}] {exc}", file=sys.stderr)
                    continue
                print(f"[{name}] ready")
            continue
        if not check_model(name, spec, project_root):
            validate_model_package(name, spec, project_root)
            print(f"[{name}] ready (already present)")
            continue
        try:
            download_model(
                name,
                spec,
                project_root=project_root,
                revision=args.revision,
                token=args.token,
                modelscope_token=args.modelscope_token,
                model_source=args.model_source,
                modelscope_repo_overrides=modelscope_repo_overrides,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            is_hf_source = spec.get("source_type", "company_hf") in {"company_hf", "official_hf"}
            if isinstance(exc, ModelScopeAccessError):
                print_modelscope_access_help(exc, args.model, args.model_source)
            elif is_hf_source and is_hf_access_error(exc):
                print_hf_access_help(name, spec, exc, args.model)
            else:
                print(f"[{name}] download failed; no later models will be attempted.", file=sys.stderr)
                print(f"Original error: {exc}", file=sys.stderr)
                print(f"Retry: python scripts/download_models.py --model {args.model}", file=sys.stderr)
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
