"""factorlib: standalone Alpha101 and GTJA191 formula library.

Every factor is one module with one public entry point:
    from factorlib.alpha.alpha001 import compute
    from factorlib.gtja.gtja_001 import compute

The library does not import aurumq_rl. Alpha101 and GTJA191 operators remain
separate because same-named operators can have different semantics.
""",
