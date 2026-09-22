#!/usr/bin/env python3
"""Fetch pinned official source repositories required by inference models."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_SOURCE_CONFIG = PROJECT_ROOT / "configs" / "sources.json"
DEFAULT_NETWORK_TIMEOUT_SECONDS = 30.0
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 1800.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_names_for_models(
    requested_model: str,
    models: dict[str, Any],
    sources: dict[str, Any],
) -> list[str]:
    model_names = list(models) if requested_model == "all" else [requested_model]
    unknown = [name for name in model_names if name not in models]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(models)}"
        )
    resolved_models: list[str] = []
    visiting: set[str] = set()

    def add_model(model_name: str) -> None:
        if model_name in resolved_models:
            return
        if model_name in visiting:
            raise ValueError(f"Cyclic model dependency involving: {model_name}")
        if model_name not in models:
            raise ValueError(f"Unknown model dependency: {model_name}")
        visiting.add(model_name)
        resolved_models.append(model_name)
        for dependency in models[model_name].get("model_dependencies", []):
            add_model(str(dependency))
        visiting.remove(model_name)

    for model_name in model_names:
        add_model(model_name)

    selected: list[str] = []
    for model_name in resolved_models:
        for source_name in models[model_name].get("source_dependencies", []):
            if source_name not in sources:
                raise ValueError(f"Model {model_name} references unknown source: {source_name}")
            if source_name not in selected:
                selected.append(source_name)
    return selected


def _run_git(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    timeout_seconds: float | None = None,
) -> str:
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")
    environment.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    command = ["git", "-c", "http.version=HTTP/1.1", *arguments]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            exc.timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return (stdout or "").strip() if capture else ""


def _clear_stale_shallow_locks(destination: Path) -> None:
    git_directory = destination / ".git"
    if not git_directory.is_dir():
        return
    locks = [git_directory / "shallow.lock"]
    locks.extend(git_directory.glob("modules/**/shallow.lock"))
    for lock in locks:
        lock.unlink(missing_ok=True)


def _clear_failed_fetch_state(destination: Path) -> None:
    """Remove unusable temporary pack state before trying another route."""
    _clear_stale_shallow_locks(destination)
    git_directory = destination / ".git"
    if not git_directory.is_dir():
        return
    patterns = (
        "objects/**/tmp_*",
        "modules/**/objects/**/tmp_*",
        "modules/**/index.lock",
    )
    for pattern in patterns:
        for path in git_directory.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def github_repository_candidates(
    repository: str,
    proxy_routes: list[dict[str, str]],
    preferred_route: str | None = None,
) -> list[tuple[str, str]]:
    github_prefix = "https://github.com/"
    if not repository.startswith(github_prefix):
        return [("upstream", repository)]

    repository_path = repository.removeprefix(github_prefix)
    candidates = [("github.com", repository)]
    for route in proxy_routes:
        name = str(route.get("name", "")).strip()
        url_prefix = str(route.get("url_prefix", "")).strip()
        if not name or not url_prefix.startswith("https://"):
            raise ValueError(f"Invalid GitHub proxy route: {route}")
        candidates.append((name, f"{url_prefix}{repository_path}"))
    if preferred_route:
        candidates.sort(key=lambda candidate: candidate[0] != preferred_route)
    return candidates


def _git_error_summary(error: BaseException) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return f"timed out after {error.timeout:g} seconds"
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or error.stdout or "").replace("\n", " ").strip()
        detail = " ".join(detail.split())
        return detail[:500] or f"git exited with status {error.returncode}"
    return str(error)


def _has_commit(destination: Path, revision: str) -> bool:
    try:
        _run_git(["cat-file", "-e", f"{revision}^{{commit}}"], cwd=destination, capture=True)
    except subprocess.CalledProcessError:
        return False
    return True


def fetch_revision_with_fallback(
    name: str,
    destination: Path,
    repository: str,
    revision: str,
    proxy_routes: list[dict[str, str]],
    network_timeout_seconds: float,
    transfer_timeout_seconds: float,
    preferred_route: str | None = None,
) -> str:
    failures: list[str] = []
    low_speed_time = max(1, int(network_timeout_seconds))
    candidates = github_repository_candidates(repository, proxy_routes, preferred_route)
    for route_name, candidate in candidates:
        print(f"[source] {name}: probe {route_name} ({candidate})", flush=True)
        try:
            _run_git(
                ["ls-remote", candidate, "HEAD"],
                capture=True,
                timeout_seconds=network_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = _git_error_summary(exc)
            failures.append(f"{route_name} probe: {detail}")
            print(f"[source] {name}: {route_name} probe failed: {detail}", file=sys.stderr)
            continue
        print(
            f"[source] {name}: fetch through {route_name} "
            f"(transfer limit {transfer_timeout_seconds:g}s)",
            flush=True,
        )
        try:
            _run_git(
                [
                    "-c",
                    "http.lowSpeedLimit=1",
                    "-c",
                    f"http.lowSpeedTime={low_speed_time}",
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    candidate,
                    revision,
                ],
                cwd=destination,
                capture=True,
                timeout_seconds=transfer_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _clear_failed_fetch_state(destination)
            detail = _git_error_summary(exc)
            failures.append(f"{route_name}: {detail}")
            print(f"[source] {name}: {route_name} failed: {detail}", file=sys.stderr)
            continue
        if not _has_commit(destination, revision):
            failures.append(f"{route_name}: fetched revision is unavailable locally")
            continue
        print(f"[source] {name}: fetched through {route_name}", flush=True)
        return route_name
    raise RuntimeError("all GitHub download routes failed; " + " | ".join(failures))


def validate_submodules(destination: Path) -> None:
    status = _run_git(
        ["submodule", "status", "--recursive"],
        cwd=destination,
        capture=True,
    )
    invalid = [line for line in status.splitlines() if line.startswith(("-", "+", "U"))]
    if invalid:
        details = "\n".join(f"  {line}" for line in invalid)
        raise RuntimeError(f"source has missing or mismatched submodules:\n{details}")


def update_submodules_with_fallback(
    name: str,
    destination: Path,
    proxy_routes: list[dict[str, str]],
    network_timeout_seconds: float,
    transfer_timeout_seconds: float,
    preferred_route: str | None = None,
) -> str:
    _run_git(["submodule", "sync", "--recursive"], cwd=destination)
    attempts: list[tuple[str, str | None]] = [("github.com", None)]
    attempts.extend(
        (str(route["name"]), str(route["url_prefix"])) for route in proxy_routes
    )
    if preferred_route:
        attempts.sort(key=lambda attempt: attempt[0] != preferred_route)
    failures: list[str] = []
    low_speed_time = max(1, int(network_timeout_seconds))
    for route_name, rewrite_prefix in attempts:
        arguments = [
            "-c",
            "http.lowSpeedLimit=1",
            "-c",
            f"http.lowSpeedTime={low_speed_time}",
        ]
        if rewrite_prefix:
            arguments.extend(
                ["-c", f"url.{rewrite_prefix}.insteadOf=https://github.com/"]
            )
        arguments.extend(
            ["submodule", "update", "--init", "--recursive", "--depth=1"]
        )
        print(f"[source] {name}: try submodules through {route_name}", flush=True)
        try:
            _run_git(
                arguments,
                cwd=destination,
                capture=True,
                timeout_seconds=transfer_timeout_seconds,
            )
            validate_submodules(destination)
        except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _clear_failed_fetch_state(destination)
            detail = _git_error_summary(exc)
            failures.append(f"{route_name}: {detail}")
            print(
                f"[source] {name}: submodules through {route_name} failed: {detail}",
                file=sys.stderr,
            )
            continue
        return route_name
    raise RuntimeError("all GitHub submodule routes failed; " + " | ".join(failures))


def _set_official_origin(destination: Path, repository: str) -> None:
    try:
        _run_git(["remote", "get-url", "origin"], cwd=destination, capture=True)
    except subprocess.CalledProcessError:
        _run_git(["remote", "add", "origin", repository], cwd=destination)
    else:
        _run_git(["remote", "set-url", "origin", repository], cwd=destination)


def validate_official_origin(destination: Path, repository: str) -> None:
    actual = _run_git(["remote", "get-url", "origin"], cwd=destination, capture=True)
    if actual != repository:
        raise RuntimeError(
            f"source origin mismatch: expected {repository}, got {actual or 'missing'}"
        )


def fetch_source(
    name: str,
    spec: dict[str, Any],
    project_root: Path,
    *,
    proxy_routes: list[dict[str, str]],
    network_timeout_seconds: float,
    transfer_timeout_seconds: float,
    preferred_route: str | None = None,
) -> str | None:
    destination = (project_root / spec["local_dir"]).resolve()
    source_root = (project_root / "third_party" / "src").resolve()
    if source_root not in destination.parents:
        raise ValueError(f"Source destination must be under {source_root}: {destination}")
    revision = str(spec["revision"])
    repository = str(spec["repo_url"])
    recursive = bool(spec.get("recursive", False))

    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(f"Refusing to replace non-Git source path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    checkout = destination
    if not destination.exists():
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.download-", dir=destination.parent)
        )
        checkout = temporary

    try:
        if temporary is not None:
            _run_git(["init", "--quiet"], cwd=checkout)
        _set_official_origin(checkout, repository)

        if temporary is None:
            current = _run_git(["rev-parse", "HEAD"], cwd=checkout, capture=True)
            if current != revision:
                dirty = _run_git(["status", "--porcelain"], cwd=checkout, capture=True)
                if dirty:
                    raise RuntimeError(
                        "Source checkout has local changes and cannot switch revisions: "
                        f"{destination}"
                    )

        selected_route: str | None = None
        if not _has_commit(checkout, revision):
            selected_route = fetch_revision_with_fallback(
                name,
                checkout,
                repository,
                revision,
                proxy_routes,
                network_timeout_seconds,
                transfer_timeout_seconds,
                preferred_route,
            )
        _run_git(["switch", "--detach", revision], cwd=checkout)

        if recursive:
            submodule_route = update_submodules_with_fallback(
                name,
                checkout,
                proxy_routes,
                network_timeout_seconds,
                transfer_timeout_seconds,
                selected_route or preferred_route,
            )
            selected_route = selected_route or submodule_route

        current = _run_git(["rev-parse", "HEAD"], cwd=checkout, capture=True)
        if current != revision:
            raise RuntimeError(f"{name} revision mismatch: expected {revision}, got {current}")
        required = checkout / str(spec["required_file"])
        if not required.is_file():
            raise RuntimeError(f"{name} source is missing required file: {required}")
        validate_official_origin(checkout, repository)

        if temporary is not None:
            checkout.rename(destination)
            temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    print(f"[source] {name} {revision} -> {destination}")
    return selected_route


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        default="all",
        help="Fetch sources required by one model key from configs/models.json, or all.",
    )
    selection.add_argument(
        "--source",
        help="Fetch one source key from configs/sources.json.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument(
        "--network-timeout",
        type=float,
        default=None,
        help="Git route probe and low-speed stall timeout in seconds.",
    )
    parser.add_argument(
        "--transfer-timeout",
        type=float,
        default=None,
        help="Wall-clock limit for one Git fetch or recursive submodule attempt.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate required files, revisions, official origins and recursive "
            "submodules without network access."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = load_json(args.model_config.resolve())["models"]
    source_config = load_json(args.source_config.resolve())
    sources = source_config["sources"]
    proxy_routes = list(source_config.get("github_proxy_routes", []))
    timeout_seconds = float(
        args.network_timeout
        if args.network_timeout is not None
        else source_config.get("network_timeout_seconds", DEFAULT_NETWORK_TIMEOUT_SECONDS)
    )
    if timeout_seconds <= 0:
        raise SystemExit("--network-timeout must be positive")
    transfer_timeout_seconds = float(
        args.transfer_timeout
        if args.transfer_timeout is not None
        else source_config.get("transfer_timeout_seconds", DEFAULT_TRANSFER_TIMEOUT_SECONDS)
    )
    if transfer_timeout_seconds <= 0:
        raise SystemExit("--transfer-timeout must be positive")
    if args.source:
        if args.source not in sources:
            raise SystemExit(
                f"Unknown source: {args.source}. Choose from: {', '.join(sources)}"
            )
        names = [args.source]
    else:
        try:
            names = source_names_for_models(args.model, models, sources)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    project_root = args.project_root.resolve()
    if not names:
        print("[source] selected model has no external source checkout")
        return 0
    failed = False
    preferred_route: str | None = None
    for name in names:
        spec = sources[name]
        destination = project_root / spec["local_dir"]
        if args.check_only:
            required = destination / spec["required_file"]
            if not (destination / ".git").is_dir() or not required.is_file():
                failed = True
                print(f"[source] {name}: missing or incomplete ({destination})")
                continue
            try:
                current = _run_git(["rev-parse", "HEAD"], cwd=destination, capture=True)
            except subprocess.CalledProcessError:
                failed = True
                print(f"[source] {name}: cannot read Git revision ({destination})")
                continue
            if current != spec["revision"]:
                failed = True
                print(
                    f"[source] {name}: revision mismatch "
                    f"(expected {spec['revision']}, got {current})"
                )
            else:
                try:
                    validate_official_origin(destination, str(spec["repo_url"]))
                    if spec.get("recursive"):
                        validate_submodules(destination)
                except (RuntimeError, subprocess.CalledProcessError) as exc:
                    failed = True
                    print(f"[source] {name}: {exc}")
                    continue
                print(f"[source] {name}: ready ({current})")
            continue
        try:
            selected_route = fetch_source(
                name,
                spec,
                project_root,
                proxy_routes=proxy_routes,
                network_timeout_seconds=timeout_seconds,
                transfer_timeout_seconds=transfer_timeout_seconds,
                preferred_route=preferred_route,
            )
            if selected_route:
                preferred_route = selected_route
        except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
            print(f"[source] {name}: fetch failed: {exc}")
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
