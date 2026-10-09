"""Scan all factor sources for private-helper and local-ops-import dependencies."""
import ast
from pathlib import Path

SRC = Path("/media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors")
for group in ("alpha101", "gtja191"):
    d = SRC / group
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_") or p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        privates = {n: f for n, f in funcs.items() if n.startswith("_")}
        for name, f in funcs.items():
            if not (name.startswith("alpha") or name.startswith("gtja")):
                continue
            called = {s.func.id for s in ast.walk(f) if isinstance(s, ast.Call) and isinstance(s.func, ast.Name)}
            used = sorted(set(called) & set(privates))
            local_imports = []
            for sub in ast.walk(f):
                if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.endswith("_ops"):
                    local_imports.extend(x.name for x in sub.names)
            if used or local_imports:
                print(f"{group}/{p.name}::{name} privates={used} local_ops_imports={local_imports}")
