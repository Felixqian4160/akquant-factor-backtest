"""Build the factorlib alpha/gtja groups from the canonical AurumQ sources.

Output: /media/felix/f/quant/akquant-factor-backtest/factorlib/{alpha,gtja}/

Run order (see REGENERATE.sh):
  1. build_factorlib_v2.py       (this file — wipes factorlib/, rebuilds alpha+gtja)
  2. build_extended_factorlib.py (adds talib + academic + github groups)
  3. verify_factorlib.py         (runs every module)

The wipe step is guarded: OUT must end with "factorlib".

Deduplication (2026-10-10): 14 GTJA191 modules are bit-exact duplicates of
Alpha101 modules on the v34 panel (see
evidence/audit_lookahead_20261010/v34_exact_dups.json). One implementation per
formula is kept — the alpha side — so the GTJA group ships 177 modules.

Dependency carrying:
- Module-level private helpers referenced by a factor (e.g. `_abs`,
  `_stock_row_index`, `_bool_lt_signed`) are carried into the emitted file,
  resolved recursively.
- Function-local `from ._ops import X` lines are stripped and X is merged
  into the module-level import from factorlib._ops.<ops_module>.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

SRC = Path("/media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors")
OUT = Path("/media/felix/f/quant/akquant-factor-backtest/factorlib")

# 2026-10-10: exact duplicates removed — each module below is bit-identical to the
# Alpha101 module on the v34 panel (see
# evidence/audit_lookahead_20261010/v34_exact_dups.json); the alpha side is kept.
DROP_DUPLICATES = {
    "gtja_007", "gtja_013", "gtja_016", "gtja_042", "gtja_083", "gtja_086",
    "gtja_095", "gtja_099", "gtja_104", "gtja_105", "gtja_107", "gtja_120",
    "gtja_139", "gtja_184",
}


def top_functions(path: Path) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def ops_import_names(tree: ast.Module) -> list[str]:
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("_ops"):
            return [x.name for x in n.names]
    return []


def collect_name_ids(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def local_ops_import_names(fn: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.endswith("_ops"):
            names |= {x.name for x in sub.names}
    return names


def resolve_private_helpers(fn: ast.FunctionDef,
                            funcs: dict[str, ast.FunctionDef]) -> list[ast.FunctionDef]:
    """Recursively locate module-level private helpers referenced by fn."""
    needed: list[ast.FunctionDef] = []
    seen: set[str] = set()
    queue: list[ast.AST] = [fn]
    while queue:
        current = queue.pop()
        for name in collect_name_ids(current):
            if name in funcs and name.startswith("_") and name not in seen:
                helper = funcs[name]
                seen.add(name)
                needed.append(helper)
                queue.append(helper)
    return needed


def write_factor_file(out_path: Path, group: str, source_path: Path,
                      fn: ast.FunctionDef, ops: list[str], ops_module: str,
                      funcs: dict[str, ast.FunctionDef]) -> None:
    source = ast.unparse(fn)
    prefix = f"def {fn.name}("
    if not source.startswith(prefix):
        raise RuntimeError(f"unexpected function source for {fn.name}")
    source = "def compute(" + source[len(prefix):]
    # Merge function-local `from ._ops import X` into the top import line.
    source = re.sub(r"^\s*from \._ops import .*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"\n{3,}", "\n\n", source)
    import_names = list(dict.fromkeys(ops + sorted(local_ops_import_names(fn))))
    import_line = ", ".join(import_names) if import_names else "TS_PART"
    helpers = resolve_private_helpers(fn, funcs)
    helper_block = ("\n\n".join("# --- private helper (from canonical source) ---\n" + ast.unparse(h)
                                for h in helpers) + "\n\n") if helpers else ""
    doc = ast.get_docstring(fn) or f"Canonical {group} factor {fn.name}."
    text = f'''"""{fn.name} — standalone {group} factor.

{doc}

Canonical source: {source_path}

Usage:
    from factorlib.{group}.{fn.name} import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.{ops_module} import {import_line}

{helper_block}{source}
'''
    out_path.write_text(text)


def build_group(group: str, source_dir: Path, out_dir: Path,
                ops_module: str) -> tuple[int, list[str]]:
    count = 0
    names: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        ops = ops_import_names(tree)
        for fn in top_functions(path):
            if not (fn.name.startswith("alpha") or fn.name.startswith("gtja")):
                continue
            if fn.name in DROP_DUPLICATES:
                continue
            write_factor_file(out_dir / f"{fn.name}.py", group, path, fn, ops, ops_module, funcs)
            names.append(fn.name)
            count += 1
    return count, sorted(names)


def main() -> None:
    if OUT.exists():
        assert OUT.name == "factorlib", f"refusing to wipe unexpected path: {OUT}"
        shutil.rmtree(OUT)
    (OUT / "_ops").mkdir(parents=True)

    alpha_ops = (SRC / "alpha101" / "_ops.py").read_text()
    gtja_ops = (SRC / "gtja191" / "_ops.py").read_text()
    gtja_ops = gtja_ops.replace(
        "from aurumq_rl.factors.alpha101._ops import (",
        "from factorlib._ops.alpha101_ops import (",
    )
    (OUT / "_ops" / "alpha101_ops.py").write_text(alpha_ops)
    (OUT / "_ops" / "gtja191_ops.py").write_text(gtja_ops)
    (OUT / "_ops" / "__init__.py").write_text(
        '"""Separate Alpha101 and GTJA191 operator namespaces."""\n'
    )

    alpha_n, alpha_names = build_group(
        "alpha", SRC / "alpha101", OUT / "alpha", "alpha101_ops"
    )
    gtja_n, gtja_names = build_group(
        "gtja", SRC / "gtja191", OUT / "gtja", "gtja191_ops"
    )

    (OUT / "alpha" / "__init__.py").write_text(
        '"""107 standalone WorldQuant Alpha101 factor modules."""\n'
    )
    (OUT / "gtja" / "__init__.py").write_text(
        '"""177 standalone GTJA191 factor modules (14 exact duplicates of Alpha101 factors removed)."""\n'
    )
    (OUT / "__init__.py").write_text('''"""factorlib: standalone Alpha101 and GTJA191 formula library.

Every factor is one module with one public entry point:
    from factorlib.alpha.alpha001 import compute
    from factorlib.gtja.gtja_001 import compute

The library does not import aurumq_rl. Alpha101 and GTJA191 operators remain
separate because same-named operators can have different semantics.
""",
''')

    manifest = {
        "source": str(SRC),
        "output": str(OUT),
        "alpha_count": alpha_n,
        "gtja_count": gtja_n,
        "total_count": alpha_n + gtja_n,
        "dropped_duplicates": sorted(DROP_DUPLICATES),
        "dropped_note": (
            "14 GTJA191 modules removed as bit-exact duplicates of Alpha101 "
            "factors (verified on the v34 panel, 2026-10-10; see "
            "evidence/audit_lookahead_20261010/v34_exact_dups.json). "
            "Alpha side kept; formulas remain recoverable from the aurumq_rl source."
        ),
        "alpha_names": alpha_names,
        "gtja_names": gtja_names,
        "operator_modules": [
            "_ops/alpha101_ops.py",
            "_ops/gtja191_ops.py",
        ],
        "entrypoint": "compute(panel: polars.DataFrame) -> polars.Series",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ("alpha_count", "gtja_count", "total_count")}, ensure_ascii=False))
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
