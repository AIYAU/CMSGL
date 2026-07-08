# HyLiOSR Compatibility Layer

This folder contains the minimal compatibility subset copied from the public
HyLiOSR repository that is required to preserve the current CC-SGCL working
behavior.

Included subset:
- `rscls`
- `make_sample`
- `gtcfm`

Purpose:
- preserve patch extraction behavior
- preserve known-class train split behavior
- preserve open-set evaluation behavior

Origin:
- public HyLiOSR codebase
- used only as a compatibility layer for experiment protocol reproduction

Important:
- This folder is third-party compatibility code, not the CC-SGCL method itself.
- The standalone CC-SGCL method implementation lives under `cc_sgcl/`.
