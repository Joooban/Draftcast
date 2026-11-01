# Draftcast Refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layered, honest explanations and scoreboard-friendly live entry (blank = unknown, 10:00 or 15:00), plus the remaining refinement-list items.

**Architecture:**
- The API answers state and both-mode requests with a logistic regression fitted on just the known fields. It fits from shipped training rows, caches the fit, and scores it on shipped test-patch rows, so every quoted number belongs to the model that predicted.
- Training adds minute-10 features and models, calibration bins, and a data summary.
- The frontend replaces sliders with typed per-player and per-team entry and renders verdict → reasons → details.

**Tech Stack:** Python 3.13, pandas, scikit-learn, FastAPI/Pydantic v2, React 19 + TypeScript + Vite, Plotly (lazy).

**Spec:** `docs/superpowers/specs/2026-11-02-draftcast-refinements-design.md`

## Global Constraints

- **No push, deploy or public-repo change.** Commits stay local (the owner said to hold off).
- **No Riot calls from the app.** The API never imports `lightgbm`, `statsmodels` or `lolwp.models.train`.
- **Blank = unknown, never 0.** No explanation lists a feature whose input value is 0.
- **The waterfall stays exact:** `baseline + Σ value_pct/100 == probability` (abs 1e-9).
- **Leakage rules stay:** training never sees the test patch, and combined-model train rows use out-of-fold `draft_logit`.
- **TDD:** every behaviour gets a test written first. Tests run without real data (synthetic pipeline in `tests/conftest.py`).
- **Feature changes:** remove `xp_diff`, leaving 13 state features. Minutes are exactly `(10, 15)`. The minute-10 population is the same as minute 15 (duration ≥ 960 s).
- **Strength words** use `max(p, 1-p)`: < 0.55 "toss-up", < 0.65 "slight edge", < 0.80 "favored", else "strong favorite".
- **Calibration:** 10 equal-width bins of predicted P(Blue); empty bins dropped.
- **Files:** written UTF-8 with LF. Don't use Python `open()` without `encoding="utf-8"`.

## Review Focus

1. **A state request with every field blank.** Expect a 422 saying at least one game-state value is needed, never a crash or a fit on zero features. *Test: Task 4 `test_state_all_blank_422`.*
2. **Both mode where only the draft plus one state field is known.** Expect the refit on `draft_logit` plus that field, an exact waterfall, and a split that sums to `probability - baseline`. *Test: Task 4 `test_both_partial_state_split_exact`.*
3. **One player's CS left blank while the other nine are typed.** Expect `cs_diff` unknown (omitted), not a partial sum. *Test: Task 6 `toState` case in `web/src/live.check.ts` (no frontend test runner, so this is a type-checked assertion script).*
4. **Fitting the same field set twice.** Expect the second request to reuse the cached fit. *Test: Task 4 `test_refit_is_cached`.*
5. **A minute-10 request with the full field set.** Expect the minute-10 models, never the minute-15 ones. *Test: Task 4 `test_minute_10_uses_m10_rows`.*

---

### Task 1: Features: drop XP, add minute-10 rows, count drop reasons

**Files:**
- Modify: `lolwp/config.py`, `lolwp/features/parse.py`, `lolwp/features/build.py`, `lolwp/api/predictor.py` (LABELS), `web/src/types.ts`
- Test: `tests/test_config.py`, `tests/test_parse.py`, `tests/test_build.py`, `tests/test_state.py`

**Interfaces:**
- Produces:
  - In `config.py`, `STATE_FEATURES` (13 names, no `xp_diff`) and `STATE_MINUTES = (10, 15)`.
  - `parse_match_checked(match: dict, platform: str | None = None) -> tuple[MatchRecord | None, str | None]`. The reason is one of `"other_platform"`, `"wrong_queue"`, `"not_10_players"`, `"early_surrender"`, `"role_missing"`, `"malformed"`. `parse_match` returns its first element.
  - `build(...)` writes `data/state.parquet` (minute 15, unchanged name), `data/state_m10.parquet` (same columns) and `artifacts/build_summary.json` = `{"dropped": {reason: n}, "patches": [sorted patch strings]}`.
  - The return dict gains `"dropped_by_reason"`. `main` prints each reason count.

- [ ] **Step 1: Write the failing tests**
  - In `test_config.py`, `len(STATE_FEATURES) == 13`, `"xp_diff" not in STATE_FEATURES`, and `STATE_MINUTES == (10, 15)`.
  - In `test_parse.py`, `parse_match_checked` returns the reason for each existing rejection test case (wrong queue → `"wrong_queue"`, early surrender → `"early_surrender"`, blank/missing/duplicate role → `"role_missing"`, OC1 id with `platform="sg2"` → `"other_platform"`, `{}` → `"malformed"`), and `(record, None)` for the valid fixture.
  - In `test_state.py`, `state_features(match, timeline, minute=10)` returns a dict without `xp_diff`. Its `gold_diff` equals the sum computed from `frames[10]`.
  - In `test_build.py::test_dataset_columns_and_filters`, extend it so that:
    - `state_m10.parquet` exists with the same columns and match IDs as `state.parquet`;
    - `build_summary.json` has `dropped` summing to the build's `dropped`;
    - `patches == ["15.19", "15.20", "15.21"]`.

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.venv/Scripts/python -m pytest tests/test_config.py tests/test_parse.py tests/test_state.py tests/test_build.py -q`
  Expected: FAIL. `parse_match_checked` doesn't exist yet, the length is 14, and `state_m10.parquet` is missing.

- [ ] **Step 3: Implement**
  - Remove `xp_diff` from `STATE_FEATURES`, `STATE_RANGES`, `LABELS`, and from `STATE_FEATURES`, `STATE_RANGES` and `STATE_LABELS` in `types.ts`.
  - In `state.py`, stop computing XP.
  - In `build`, compute `state_features(details, timeline, m)` for each `m` in `STATE_MINUTES` while streaming, keeping `states: dict[int, dict[str, dict]]`.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `.venv/Scripts/python -m pytest -q`
  Expected: PASS everywhere except the `[real]`-parametrized tests. Those load the stale real artifacts until Task 2 retrains, so mark them with Step 5's rerun.

- [ ] **Step 5: Commit**

  ```bash
  git add lolwp tests web/src/types.ts
  git commit -m "feat: drop XP, add minute-10 state rows, count drop reasons"
  ```

### Task 2: Training: shared helpers, minute-10 models, calibration, shipped rows

**Files:**
- Create: `lolwp/models/common.py`
- Modify: `lolwp/models/train.py`, `lolwp/api/predictor.py` (import `COMBINED_FEATURES` from common)
- Test: `tests/test_models_sanity.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: Task 1 parquet files and `build_summary.json`.
- Produces (in `lolwp/models/common.py`, importing only numpy, pandas and sklearn):
  - `COMBINED_FEATURES`, `make_lr()`, `scores(y, p) -> dict`;
  - `calibration_bins(y, p, n_bins=10) -> list[{"lo": float, "hi": float, "mean_pred": float, "win_rate": float, "n": int}]`;
  - `SHIPPED_COLUMNS = ["split", "blue_win", "draft_logit", *STATE_FEATURES]`.

  `train.py` re-exports `COMBINED_FEATURES`, `make_lr` and `scores` from common.
- Produces (artifacts):
  - `state_m10_model.joblib`, `combined_m10_model.joblib`;
  - `state_rows_m10.parquet`, `state_rows_m15.parquet` (exactly `SHIPPED_COLUMNS`; `draft_logit` is out-of-fold on train rows and from the full-train draft model on test rows; no match IDs);
  - `metrics.json` gains:
    - `models["state_lr_m10"]` and `models["combined_lr_m10"]`;
    - `"calibration": {model_name: bins}` for `draft_lr`, `state_lr`, `combined_lr`, `state_lr_m10`, `combined_lr_m10`;
    - `"data": {"train_patches": [...], "test_patch": str, "n_games": int, "dropped": {...}}`.

- [ ] **Step 1: Write the failing tests**
  - In `test_models_sanity.py`:
    - `test_metrics_file_complete` also requires the new model names, a `calibration` entry per app model where `sum(b["n"]) == n_test` for that model, and `data.n_games == n_train.draft + n_test.draft`.
    - New `test_shipped_rows_have_no_ids`: both parquet files have exactly `SHIPPED_COLUMNS`, and both splits are present.
    - New `test_calibration_bins_basic`: `calibration_bins([0,1,1,0],[0.05,0.95,0.92,0.15])` gives two bins, `(0.0, 0.1)` with n=1 and `(0.9, 1.0)` with n=2, win_rate 1.0.
  - In `test_api.py`, new `test_api_does_not_import_training`. It runs a subprocess `python -c "import sys; from lolwp.api.main import create_app; create_app('<synthetic artifacts>'); print(any(m in sys.modules for m in ('lightgbm','statsmodels','lolwp.models.train')))"` and asserts stdout is `False`.

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.venv/Scripts/python -m pytest tests/test_models_sanity.py tests/test_api.py -q -k "metrics or shipped or calibration or import_training"`
  Expected: FAIL. `common` doesn't exist yet, the keys are missing, and `True` is printed.

- [ ] **Step 3: Implement** the helpers in `common.py`. In `train.py`, loop the state and combined LR over `STATE_MINUTES`. Minute 15 keeps the existing model names, file names and both LGBM benchmarks. Minute 10 gets LR only, under the `_m10` suffix.

- [ ] **Step 4: Run the full suite and retrain on the real data**

  Run:
  ```
  .venv/Scripts/python -m pytest -q
  .venv/Scripts/python -m lolwp.features.build
  .venv/Scripts/python -m lolwp.models.train
  .venv/Scripts/python -m lolwp.models.insights
  .venv/Scripts/python -m pytest -q
  ```
  Expected: both suite runs PASS, the second one including `[real]`.

- [ ] **Step 5: Commit** the code, tests and regenerated `artifacts/`.

  ```bash
  git commit -m "feat: minute-10 models, calibration bins, shipped rows; API no longer imports training"
  ```

### Task 3: Collector: `--limit` as a total, one-command refresh

**Files:**
- Modify: `lolwp/riot/collect.py`
- Create: `lolwp/refresh.py`
- Test: `tests/test_collect.py`, `tests/test_refresh.py`

**Interfaces:**
- Produces:
  - `collect_ids`, `collect_details` and `collect_timelines` treat `limit` as a store total. Totals are `store.counts()["match_ids"|"details"|"timelines"]`. They do nothing when it's already reached, and `collect_ids` stops between players once reached. Each takes `raise_expired: bool = False`; when True it re-raises `KeyExpiredError` instead of printing.
  - `lolwp.refresh.refresh(client, store, settings, ids: int, details: int, timelines: int, data_dir, artifacts_dir) -> bool` runs the three collect steps, then build (with `settings.platform`), train and insights. On key expiry it prints `EXPIRED_MSG`, skips the build and returns False.
  - `main()` uses argparse with `--ids`, `--details` and `--timelines` (defaults 20000, 20000, 8000) and exits 1 when `refresh` returns False.

- [ ] **Step 1: Write the failing tests**
  - `test_details_limit_is_total`: the store already holds 3 details out of 5 IDs. `collect_details(..., limit=4)` fetches exactly 1, so the FakeClient is called once.
  - `test_ids_stop_at_total`: 3 players each return 2 new IDs. `limit=3` processes 2 players, and the third stays todo.
  - `test_refresh_stops_on_expiry`: the client raises `KeyExpiredError` on the first call. `refresh` returns False, and no `draft.parquet` is written in `tmp_path`.
  - `test_refresh_runs_pipeline`: a synthetic store from `fill_store` (all fetched) with a client that's never called. `refresh(...)` returns True, and `metrics.json` exists.

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.venv/Scripts/python -m pytest tests/test_collect.py tests/test_refresh.py -q`
  Expected: FAIL. The calls exceed the totals, and `lolwp.refresh` isn't found.

- [ ] **Step 3: Implement** as specified. Existing `test_collect.py` cases must still pass unchanged.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `.venv/Scripts/python -m pytest -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git commit -m "feat: --limit is a store total; add lolwp.refresh one-command pipeline"
  ```

### Task 4: API: unknown fields, refit, layered response, meta endpoint

**Files:**
- Create: `lolwp/models/refit.py`, `lolwp/api/explain_text.py`
- Modify: `lolwp/api/schemas.py`, `lolwp/api/predictor.py`, `lolwp/api/main.py`
- Test: `tests/test_api.py`, `tests/test_refit.py`, `tests/test_explain_text.py`

**Interfaces:**
- Consumes: Task 2 artifacts and `common.py`.
- Produces (`refit.py`):
  - `@dataclass FittedModel(model, features: list[str], auc: float, accuracy: float, calibration: list[dict])`.
  - `class ModelBank(artifacts_dir)` with `get(kind: Literal["state","combined"], minute: int, features: tuple[str, ...]) -> FittedModel`:
    - the full feature set loads the saved joblib, with numbers from `metrics.json`;
    - any other set fits `make_lr()` on the shipped train rows and scores it on the shipped test rows;
    - results go in an LRU cache (`maxsize=256`) keyed by `(kind, minute, features)`;
    - `fits` is a counter, for tests.
- Produces (`schemas.py`):
  - `PredictRequest(mode, minute: Literal[10, 15] = 15, blue, red, state: dict[str, StrictInt | None] | None)`. A missing key or `null` means unknown.
  - `ContributionOut` gains `value: float | None` (the raw input; `None` for derived draft features) and `logit: float`.
  - `PredictResponse` gains:
    - `verdict: str`;
    - `strength: Literal["toss-up","slight edge","favored","strong favorite"]`;
    - `reasons: list[Reason]`, where `Reason(text: str, value_pct: float)`;
    - `omitted: list[str]`;
    - `calibration: CalibrationBin | None`, where `CalibrationBin(lo, hi, win_rate, n)`;
    - `reliability: Reliability`, where `Reliability(auc: float, accuracy: float, text: str)`;
    - `minute: int`.
- Produces (`explain_text.py`, pure functions):
  - `strength(p: float) -> str`;
  - `reason_text(feature: str, value: float, minute: int) -> str | None`. It returns None when `value == 0`. Copy rule: name the leading side and the absolute value with the unit, in a full sentence, for example:
    - "Blue leads by 2,300 gold"
    - "Red has 1 more dragon"
    - "Blue is 3 levels ahead in total"
    - "Blue took first blood"
    - "Blue's mid matchup has won 54% of games"
    - "The draft leans Blue"

    Singular and plural must agree.
  - `verdict(p: float, top_reason: str | None) -> str`:
    - "Toss-up: Blue 52%" under 0.55;
    - otherwise "{Side} is {strength}: {pct}%" plus ", mostly because {top_reason lowercased-first}" when given.
- Produces (`main.py`): `GET /api/meta` returns `metrics["data"]` plus `{"minutes": [10, 15]}`.

- [ ] **Step 1: Write the failing tests**
  - `test_explain_text.py`:
    - `strength` boundaries `0.549 → "toss-up"`, `0.55 → "slight edge"`, `0.30 → "favored"`, `0.81 → "strong favorite"`;
    - `reason_text("gold_diff", -2300, 15) == "Red leads by 2,300 gold"`;
    - `reason_text("dragons_diff", 1, 15) == "Blue has 1 more dragon"`;
    - `reason_text("dragons_diff", -2, 15) == "Red has 2 more dragons"`;
    - `reason_text("kills_diff", 0, 15) is None`;
    - `reason_text("first_blood_blue", -1, 15) == "Red took first blood"`;
    - `verdict(0.68, "Blue leads by 2,300 gold") == "Blue is favored: 68%, mostly because blue leads by 2,300 gold"`.
  - `test_refit.py` (against `synthetic_dirs`):
    - `test_full_set_uses_saved_model` (`fits == 0`);
    - `test_refit_is_cached` (two `get` calls with the same subset → `fits == 1`);
    - `test_refit_scored_on_test_rows` (`0.5 <= auc <= 1`, `sum(b["n"] for b in calibration) == ` number of test rows);
    - `test_minute_10_uses_m10_rows` (the full set at minute 10 loads `state_m10_model.joblib`).
  - `test_api.py`:
    - Update `STATE` to the 13 features.
    - `test_state_partial_ok`: `{"gold_diff": 2500, "kills_diff": 3}` → 200, `omitted` = the other 11 sorted, and contributions only for gold and kills.
    - `test_state_all_blank_422`.
    - `test_both_partial_state_split_exact`.
    - `test_bool_rejected_422`: `{"gold_diff": True}` → 422.
    - `test_reasons_skip_zero_values`: no reason text for a feature whose input is 0.
    - `test_response_has_layers`: `verdict` is non-empty, `calibration.lo <= probability <= calibration.hi`, `reliability.text` is non-empty.
    - `test_unseen_champion_ok`: copy `synthetic_dirs.artifacts_dir` to `tmp_path` and add champion id 9999 to `champions.json` in the same format as the existing entries. A draft request using it returns 200, and `DraftStats.champ_rate(9999, "top") == 0.5`. Use the real signature from `draft_stats.py`.
    - `test_meta`: `/api/meta` has `test_patch`, `n_games` and `minutes == [10, 15]`.
  - Replace `test_state_missing_feature_422`, since missing keys now mean unknown.

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.venv/Scripts/python -m pytest tests/test_explain_text.py tests/test_refit.py tests/test_api.py -q`
  Expected: FAIL. The modules don't exist yet, and the new fields are missing.

- [ ] **Step 3: Implement**
  - **Reliability text:**
    - Draft: `f"Draft-only predictions are close to a coin flip (test AUC {auc:.2f}); read this as a lean, not a call."`.
    - State/both: `f"This model was right {acc:.0%} of the time on patch {test_patch} (AUC {auc:.2f})."`, plus, when `omitted`, `f" It leaves out {n} unknown field(s); the full model is right {full_acc:.0%}."`.
  - **Calibration:** the bin from the predicting model's bins that contains `probability`. Use `None` if no bin contains it.
  - **Reasons:** up to 5 contributions in waterfall order whose `reason_text` is not None.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `.venv/Scripts/python -m pytest -q`
  Expected: PASS, including `[real]`.

- [ ] **Step 5: Commit**

  ```bash
  git commit -m "feat: unknown fields refit, layered explanation response, /api/meta"
  ```

### Task 5: Slim requirements

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `README.md` (install lines)
- Test: `tests/test_requirements.py`

**Interfaces:** none.

- [ ] **Step 1: Write the failing test** `test_runtime_requirements_minimal`:
  - `requirements.txt` names exactly fastapi, uvicorn, pydantic, pandas, pyarrow, numpy, scikit-learn, joblib, python-dotenv and requests, each pinned with `==`;
  - `requirements-dev.txt` starts with `-r requirements.txt` and contains lightgbm, statsmodels and pytest.
- [ ] **Step 2: Run it.** Expected: FAIL (136 lines).
- [ ] **Step 3: Implement**
  - Pins for the runtime list: fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.2, pandas 3.1.0, pyarrow 26.0.0, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0, python-dotenv 1.2.4, requests 2.34.2.
  - Dev file: `-r requirements.txt` plus every other line of the old freeze.
  - README: users install `requirements-dev.txt` to collect/train/test; Render installs `requirements.txt`.
- [ ] **Step 4: Verify.** In a scratch venv, `pip install -r requirements.txt` succeeds, then `python -c "from lolwp.api.main import create_app; create_app('artifacts')"` succeeds. Then run the full suite in `.venv` and expect it to PASS.
- [ ] **Step 5: Commit**

  ```bash
  git commit -m "build: runtime-only requirements.txt, dev deps in requirements-dev.txt"
  ```

### Task 6: Frontend: live entry, minute selector, side toggle, keyboard, stale results

**Files:**
- Create: `web/src/live.ts`, `web/src/live.check.ts`
- Rewrite: `web/src/components/StateInput.tsx`
- Modify: `web/src/pages/Predict.tsx`, `web/src/types.ts`, `web/src/index.css`, `web/package.json` (script `check:live`)

**Interfaces:**
- Consumes: the Task 4 request shape.
- Produces:
  - In `live.ts`:
    - `PlayerStats = { level: string; cs: string; gold: string }`;
    - `LiveEntry = { players: Record<'blue'|'red', Record<Role, PlayerStats>>; team: Record<'blue'|'red', Record<TeamStat, string>>; firsts: Record<FirstStat, 'blue'|'none'|'red'|'unknown'> }`, where `TeamStat = 'kills'|'dragons'|'towers'|'grubs'|'plates'|'wards_placed'|'wards_killed'`;
    - `emptyEntry(): LiveEntry` (all `''`, firsts `'unknown'`);
    - `toState(e: LiveEntry): Partial<Record<StateFeature, number>>` (keys only for known features).

  Summing rule:
  - `level_diff`, `cs_diff` and `gold_diff` are known only if all 10 player boxes for that stat parse as non-negative integers;
  - team stats are known only if both boxes parse;
  - firsts map blue → 1, none → 0, red → −1, unknown → absent.
  - Commas and spaces are ignored when parsing.
- **Review Focus 3:** `live.check.ts` asserts the `toState` cases and is type-checked by `tsc -b`. `npm run check:live` runs it with `node --experimental-strip-types`.

  Cases:
  - all blank → `{}`;
  - nine CS filled → no `cs_diff`;
  - all levels 9 vs 8 → `level_diff: 5`;
  - kills `'7'`/`'4'` → `kills_diff: 3`;
  - gold `'1,200'` parses to 1200.

  It throws on mismatch.

- [ ] **Step 1: Write `live.check.ts`**, then run `npm run check:live`. Expected: FAIL, because `live.ts` is missing.
- [ ] **Step 2: Implement `live.ts`.** Run `npm run check:live`. Expected: prints `live ok`.
- [ ] **Step 3: Rewrite `StateInput`**
  - Props: `{ entry: LiveEntry; onChange(e: LiveEntry): void; minute: 10 | 15; onMinute(m): void }`.
  - Minute selector: segmented buttons, "At 10:00 / At 15:00".
  - Player table: rows = roles, columns = Blue level/CS/gold and Red level/CS/gold.
  - Team table: Blue | Red per team stat.
  - Firsts: 4-option segments `? / Blue / None / Red`.
  - Every input is `type="text" inputMode="numeric"` with an `aria-label` such as "Blue mid CS".
  - Tab order runs row by row.
  - The sliders are removed.
- [ ] **Step 4: Update `Predict.tsx`**
  - Wrap the inputs in a `<form onSubmit>` so Enter predicts when `ready`.
  - Add a "Reset game state" button (`type="button"`) that calls `setEntry(emptyEntry())`.
  - Clear `result` and `error` in every input setter (draft picks, entry, minute, mode).
  - Add an "I'm on: Blue / Red / Neither" toggle, held in state and passed to `ResultPanel` as `side`.
  - State mode is ready only when `Object.keys(toState(entry)).length > 0`.
- [ ] **Step 5: Verify.** Run `npm run build` and `npm run lint` in `web/`. Expected: the build passes, with no lint warnings in the changed files.
- [ ] **Step 6: Commit**

  ```bash
  git commit -m "feat(web): typed per-player live entry, 10/15 minute, side toggle, Enter/Reset"
  ```

### Task 7: Frontend: layered result panel

**Files:**
- Rewrite: `web/src/components/ResultPanel.tsx`
- Modify: `web/src/components/WaterfallChart.tsx` (labels), `web/src/types.ts`, `web/src/index.css`

**Interfaces:**
- Consumes: the Task 4 response and the Task 6 `side` prop (`'blue'|'red'|null`).

- [ ] **Step 1: Implement**, top to bottom:
  1. **Verdict:** a large `verdict` line.
     - When `side` is set, add a line under it: "Your team ({Blue|Red}): {pct}%".
     - Draft mode shows the `reliability.text` caveat inline in muted text.
  2. **"Why":** `reasons` as a list, each `text` followed by its signed `value_pct` formatted `+11 pts`.
  3. **Detail toggle:** `<details>` with the summary "Show the full reasoning", containing:
     - **Waterfall.** Contributions whose input value is 0 get the label suffix " (even)".
     - **Factor table:** factor, your input (`value`, or "—"), effect (pts) and log-odds (`logit`, 2 dp).
     - **Both mode:** one line, "Draft moved it {+x} pts, game state {+y} pts". Remove the old two-number split bar.
     - **Reality check:** "When this model said {lo}–{hi}%, Blue won {win_rate}% of {n} test games."
     - **Reliability:** `reliability.text`.
     - **Left out:** when `omitted` is non-empty, "Left out (unknown): …", using `STATE_LABELS`.
- [ ] **Step 2: Verify.** Run `npm run build` and `npm run lint`. Expected: PASS.
- [ ] **Step 3: Commit**

  ```bash
  git commit -m "feat(web): layered result: verdict, reasons with numbers, full reasoning"
  ```

### Task 8: Insights: calibration chart, champion table, trained-on label

**Files:**
- Modify: `lolwp/models/insights.py` (`champion_rates` min games 10; add `"calibration"` and `"data"` copied from metrics.json)
- Modify: `web/src/pages/Insights.tsx`, `web/src/pages/InsightsCharts.tsx`, `web/src/components/Footer.tsx`, `web/src/api.ts` (`getMeta`), `web/src/types.ts`
- Test: `tests/test_insights.py`

- [ ] **Step 1: Write the failing test.** The insights JSON has `calibration` with the 5 app-model keys and `data.test_patch`, and its champion min-games threshold is 10.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement**
  - **Backend:** add the keys and the threshold.
  - **Calibration chart:** predicted mean vs actual win rate for `state_lr`, `combined_lr` and `state_lr_m10`, with a y = x reference line.
  - **Champion table:** replaces the dot plot. Columns are name, games, win rate and 95% CI, with:
    - a search box;
    - sort buttons on win rate / games / name;
    - a minimum-games number input (default 30).
  - **Footer:** calls `getMeta()` and shows "Trained on patches {first}–{last}, tested on {test} ({n} games)", or nothing if the call fails.
- [ ] **Step 4: Verify.** Run `.venv/Scripts/python -m pytest -q` and expect PASS. Run `npm run build` and `npm run lint` and expect PASS. Then rerun `python -m lolwp.models.insights` on the real data.
- [ ] **Step 5: Commit** the code plus `artifacts/insights.json`.

  ```bash
  git commit -m "feat: calibration chart, sortable champion table, trained-on footer"
  ```

### Task 9: Copy, docs and final pass

**Files:** `web/src/pages/Method.tsx`, `README.md`, `notebooks/01_eda.ipynb`, `notebooks/02_models.ipynb`, `notebooks/03_stats.ipynb`, `web/src/index.css`

- [ ] **Step 1: Method page**
  - Replace "Tuning uses 5-fold cross-validation inside the training set only" with "5-fold cross-validation inside the training set is used only to report how stable each model is; nothing is tuned."
  - Add short sections covering: minute 10 and 15; unknown fields → model refit on the known fields with its own test numbers; the calibration reality check; XP left out because players can't see it.
- [ ] **Step 2: README**
  - Refresh all numbers from the new `metrics.json` (13 features, minute-10 rows).
  - Add `python -m lolwp.refresh` and `requirements-dev.txt`.
  - Describe live entry.
- [ ] **Step 3: Notebooks.** Under each interpretation prompt, add a markdown cell beginning `*Draft — rewrite in your own words.*` with 2–4 sentences grounded in the real numbers. Write with `encoding="utf-8"` and LF.
- [ ] **Step 4: Layout pass (item 24)**
  - Check that the new tables collapse to stacked cards below 600px (CSS media query) or scroll inside their own container, with no page-level horizontal scroll.
  - Check that inputs have visible `:focus-visible` outlines and that every new colour uses existing tokens, so dark mode is covered.
  - Run `npm run build` and `npm run lint`.
- [ ] **Step 5: Final verification**

  Run: `.venv/Scripts/python -m pytest -q`, then `npm run build`.
  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git commit -m "docs: Method/README for live entry and refit; draft notebook interpretations"
  ```
