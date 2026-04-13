# Phase Tracker

Quick-reference for what's been done and what's next in the restructure plan.
See `plan.md` in the project root for the full detailed plan.

## Phase Status

| Phase | Name | Status | Key Changes |
|-------|------|--------|-------------|
| 0 | Cleanup & Directory Restructure | ✅ Complete | Removed `src/baselines/`, legacy scripts, `Research_Paper.md`, `results/archive/`, `.DS_Store` files. Cleaned up `tasks.py`, `routes.py`, `schemas.py`, `__init__.py` |
| 1 | Config & Hardcode Elimination | ✅ Complete | Added `reward_weights`, `charging`, `wake_cooperation` config sections. Changed `nodes` default from hardcoded 550 to `None` (resolved from config at runtime). All `550` references eliminated |
| 2 | Environment: Coverage, Charging, SoH, Cooperative Wake-Up | 🔲 Pending | See `specs/m2m-environment.md` |
| 3 | Training Output Contract (JSON + 4-Panel Visualization) | 🔲 Pending | See `specs/training-output.md` |
| 4 | Frontend Alignment | 🔲 Pending | Remove baseline UI, add DDQN-vs-DQN comparison card |
| 5 | Scripts & Post-Training Comparison | 🔲 Pending | New `scripts/compare.py` for automated comparison |
| 6 | Tests, Docs, Final Pass | 🔲 Pending | Update all test assertions, `.claude/rules/`, `CLAUDE.md` files |

## Files Deleted (Phase 0)

- `src/baselines/` (entire package)
- `tests/test_baselines.py`
- `scripts/evaluate_baselines.py`
- `scripts/generate_report.py`
- `scripts/migrate_legacy_runs.py`
- `Research_Paper.md`
- `results/archive/`
- All `.DS_Store` files

## Files Modified (Phase 0-1)

| File | Changes |
|------|---------|
| `backend/tasks.py` | Dropped baseline imports/functions; nodes resolved from config |
| `backend/routes.py` | Removed `/api/evaluate` route; added `_apply_config_defaults` |
| `backend/schemas.py` | Removed `EvaluationRequestSchema`; `nodes` default → `None` |
| `src/__init__.py` | Removed baseline re-exports |
| `config/config.yaml` | Added `reward_weights`, `charging`, `wake_cooperation` sections |
| `config/settings.py` | Added `RewardWeightsConfig`, `ChargingConfig`, `WakeCooperationConfig` dataclasses |
| `scripts/train.py` | Renamed from `train_model.py`; updated docstring |
| `setup.py` | Removed `wsn-eval` entry point |

## Known Residuals (to fix in later phases)

- `wsn_env.py` still uses hardcoded reward weights `(10, 5, 1, 2)` — Phase 2 will wire from config
- `frontend/` still references `/api/evaluate` and baselines — Phase 4 will fix
- `CLAUDE.md` (root) and `backend/CLAUDE.md` still reference baselines/evaluate — Phase 6 will update
- `frontend/CLAUDE.md` still references `_benchmarkTasks` and benchmark logic — Phase 4 will fix
