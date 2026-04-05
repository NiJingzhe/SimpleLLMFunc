"""
Unified generator for manual agent flow graphs.

Discovery:
  - Scan Python files under `--scan-root`.
  - Find functions that have the attribute `__manual_agent_flow_graph__`.
  - Each such function must be a zero-arg function returning AgentFlowGraph.

Export:
  - For each discovered function `fn`, create:
      <output_dir>/<fn.__name__>/
  - Save:
      graph.mmd
      graph.dot
      graph.png  (via Graphviz `dot` if available)

No AST/static parsing:
  - We rely on runtime annotation only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


MANUAL_ATTR_KEY = "__manual_agent_flow_graph__"


def _load_module_from_path(path: Path) -> ModuleType:
    """
    Load module by file path to avoid importing the SimpleLLMFunc package
    (which might pull extra runtime dependencies).
    """
    h = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_manual_agent_flow_mod_{path.stem}_{h}"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from path: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _iter_python_files(scan_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in scan_root.rglob("*.py"):
        if p.name.startswith("_"):
            continue
        files.append(p)
    return files


def _export_graph(graph: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mmd_path = out_dir / "graph.mmd"
    dot_path = out_dir / "graph.dot"
    png_path = out_dir / "graph.png"

    graph.save_mermaid(mmd_path)
    graph.save_dot(dot_path)

    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
        )
        print(f"[OK] PNG: {png_path}")
    except Exception as exc:
        print(f"[WARN] PNG export skipped/failed: {exc}")

    print(f"[OK] Mermaid: {mmd_path}")
    print(f"[OK] DOT: {dot_path}")


def _build_output_root(scan_root: Path, output_root: Optional[Path]) -> Path:
    if output_root is not None:
        return output_root
    return scan_root / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    default_scan_root = str(Path(__file__).resolve().parent.parent / "examples")
    parser.add_argument(
        "--scan-root",
        type=str,
        default=default_scan_root,
        help="Root directory to scan for manual graph builders. Default: ./examples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write generated graphs. Default: <scan-root>/artifacts",
    )
    args = parser.parse_args()

    # Ensure project root is on sys.path so `from SimpleLLMFunc import ...` works
    # when we import example modules by file path.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    scan_root = Path(args.scan_root).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else None

    out_root = _build_output_root(scan_root, output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning: {scan_root}")
    py_files = _iter_python_files(scan_root)
    exported = 0

    for py in py_files:
        # Lightweight pre-filter to avoid importing unrelated heavy examples.
        try:
            content_preview = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content_preview = ""
        if (MANUAL_ATTR_KEY not in content_preview) and ("manual_agent_flow_graph" not in content_preview):
            continue

        try:
            mod = _load_module_from_path(py)
        except Exception as exc:
            print(f"[WARN] Skip module {py} due to import error: {exc}")
            continue

        for _, member in inspect.getmembers(mod, inspect.isfunction):
            if not hasattr(member, MANUAL_ATTR_KEY):
                continue

            try:
                graph = member()
            except TypeError as exc:
                print(f"[WARN] {member.__name__} requires args, skip. Error: {exc}")
                continue
            except Exception as exc:
                print(f"[WARN] {member.__name__} build failed: {exc}")
                continue

            fn_out_dir = out_root / member.__name__
            print(f"[INFO] Export graph for: {member.__name__} -> {fn_out_dir}")
            _export_graph(graph, fn_out_dir)
            exported += 1

    print(f"[DONE] Exported {exported} graph(s) to: {out_root}")


if __name__ == "__main__":
    main()

