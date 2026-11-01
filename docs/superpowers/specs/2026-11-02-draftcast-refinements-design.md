# Draftcast refinements — design

Date: 2026-11-02. Builds on the merged win predictor (`docs/superpowers/plans/2026-10-30-lol-win-predictor.md`).

## Intent

Draftcast predicts who wins a ranked SG2 game from the draft, the game state at a fixed minute, or both. Two readers
use the result, and the owner said "a bit of both":

- **A player mid-game** who presses Tab around the 10- or 15-minute mark, types what they can see, and wants an
  answer they can read in a second.
- **A portfolio visitor** who wants to see how the model reasons and how far to trust it.

What the owner asked for directly:
- explanations of the prediction in layers: a short one and a deeper one, with reasoning *and* numbers;
- inputs that can be typed from the scoreboard on the fly;
- per-player entry (levels, CS) with **the app adding them up**.

Nothing should need numbers a player can't see.

## Goals

1. A layered result: verdict line → short plain-language explanation with real numbers → expandable detailed view.
2. Live-game entry from what the scoreboard shows, with blank fields allowed and handled honestly.
3. The remaining refinement-list items (below), so the app is ready to go public when the owner decides.

## Non-goals

- Deploying, pushing publicly or making the repo public. **The owner said to hold off.** Commits stay local until
  asked. (Riot registration is already done.)
- Reading the game automatically (no Live Client API, overlays or screen capture). Entry stays manual.
- Vision score: Riot's timeline data has no vision score, so it can't be a feature. Wards placed and cleared stay
  as optional fields.
- Predicting at 20 minutes: stored timelines are trimmed to 16 frames, so only minutes ≤ 15 can be modelled without
  recollecting.

## Anti-goals (fails even if it "works")

- An explanation that credits a factor the user left at even or blank.
- Treating a blank field as "even". A blank must mean "unknown", never "0".
- Confident language on draft-only predictions, which are close to a coin flip (test AUC 0.529).
- Explanation numbers that don't add up to the shown probability. The waterfall must stay exact.
- Any call to Riot from the public app.

## Constraints

- Keep the existing stack: FastAPI with the `create_app` factory, React 19 / TypeScript / Vite, Plotly lazy-loaded,
  and logistic regression as the app's model.
- Training never sees the test patch. The time-based split and out-of-fold draft stats rules stay as they are.
- Tests run without real data, via the synthetic pipeline. Every new behaviour gets a test written first.
- No new runtime dependencies unless needed. The API process must not import LightGBM or the training code.

## The how: decided with the owner

### Live entry (the owner's choices)
- **Per player:** each team has five rows (Top, Jungle, Mid, Bot, Support) with **level** and **CS**, plus an
  optional **gold** per player. The app sums each team and computes Blue − Red.
- **Per team:** **kills**, **dragons**, **towers**, **voidgrubs**, **plates**, plus optional **wards placed** and
  **wards cleared**.
- **Firsts:** first blood, dragon and tower as Blue / None / Red.
- **Blank fields are allowed.** If a field is blank for *either* team (or any player in a summed field), that
  feature is unknown.

### Layered result (the owner's choices, wording mine)
1. **Verdict line**, e.g. "Blue is favored: 68%, mostly from the 2,300 gold lead."
2. **Short explanation:** the 3–5 largest factors in plain language, each with the real input and its effect in
   percentage points. For example, "Blue leads by 2,300 gold: +11 pts" or "Red has one more dragon: −4 pts".
   Factors with value 0 are never listed.
3. **Detailed view** (collapsed by default, "Show the full reasoning"):
   - the full waterfall;
   - every factor in a table: input, effect, and the same in log-odds for data readers;
   - the draft vs game-state split in Both mode;
   - a **reality check** (below);
   - a **reliability note** for the mode.

## The how: my calls (owner may override)

- **Unknown fields → a model without them** (see "Settled after the builder check"). The API fits a logistic regression on just the known features, for
  that minute and mode, from shipped training rows. Results are cached by (mode, minute, feature set).
  - Fits take well under a second on about 17k rows.
  - The saved full models serve inputs where every field is filled.
  - The response says which fields were left out.
- **Minute selector: 10:00 or 15:00.** `build` also writes minute-10 state features, and `train` trains and
  reports the minute-10 state and combined models next to the minute-15 ones.
- **Reality check = calibration of this prediction.** "Among test-patch games where the model said 65–70%, Blue won
  68% (n = 512)." It comes from per-model calibration bins computed at train time. When gold is known, a second line
  uses the gold curve: "teams with about this gold lead won X%".
- **Reliability note per mode,** taken from metrics.json. For example, Draft: "Draft-only predictions are close to a
  coin flip (test AUC 0.53); treat 54% as 'slight lean'." Game state: "Right about 74% of the time on the newest
  patch."
- **Plain-language strength words** on the verdict: under 55% "toss-up", 55–65% "slight edge", 65–80% "favored",
  over 80% "strong favorite".
- **"I'm on" toggle (Blue / Red / Neither):** the result reads as "your team" when set. It's display only; the
  request never changes.
- **Notebook interpretations:** I write a draft under each prompt, marked *"Draft — rewrite in your own words"*,
  because a portfolio reader will take them as the owner's analysis.

### Settled after the builder check (my calls)
- **XP is dropped from the state features.** Players can't see it, and total level carries almost the same
  signal. All state and combined models are retrained without it, leaving 13 state features. The README and Method
  numbers are refreshed from the new metrics.
- **The refit is the main path, not a fallback.** Gold and wards are optional, so most live predictions leave
  something out. Shipped rows (`artifacts/state_rows_m{10,15}.parquet`) hold, for each row:
  - the state features;
  - `blue_win`;
  - `split` (train or test);
  - the out-of-fold `draft_logit` for train rows, and the full-stats one for test rows. No match IDs.
- **Refit models get their own honest numbers.** Each refit is trained on train rows only. Its test AUC, accuracy
  and calibration bins are computed on the shipped test-patch rows when it's fitted, then cached with the model. The
  reliability note and the reality check always describe the model that actually made the prediction, never the
  full model's better numbers.
- **Both mode with blanks** refits the combined model on `draft_logit` plus the known state fields. The split is
  worked out the same way as now.
- **The firsts can be unknown.** Each one has four choices: Blue / None / Red / ?. "?" is the default. Every
  game-state field starts blank (unknown), so nothing is ever assumed to be even.
- **The minute-10 population matches minute 15** (games ≥ 16 min). This keeps the two directly comparable; games
  ending between 10 and 16 minutes are almost all remakes or early surrenders, which are dropped anyway.

## Remaining refinement-list items (all in scope)

| # | Item | Expected behaviour |
|---|---|---|
| 1 | Lazy API imports | `lolwp.api` never imports `lightgbm` or `lolwp.models.train`. Shared helpers move to a small module. A test asserts `"lightgbm" not in sys.modules` after `create_app`. |
| 2 | Slim requirements | `requirements.txt` holds only the runtime needed by `render.yaml`'s build and start commands, with pins. `requirements-dev.txt` holds everything else (training, notebooks, tests). The README says which to install. |
| 5 | Method wording | Remove "Tuning uses 5-fold CV". Say CV is used for reporting. Add the minute-10 model, unknown-field handling and calibration. |
| 10 | Keyboard | Enter in any input runs Predict when inputs are valid. A Reset button clears game-state entries. |
| 11 | Stale result | Any input change (draft, state, minute, mode) clears the shown result and error. |
| 12 | Draft weakness | Covered by the reliability note and the strength words. |
| 13 | Both-mode labels | One clear split: "Draft moved it +X pts, game state +Y pts". The draft-only probability isn't shown in Both mode. |
| 14 | Reason text | Zero-valued features are never reasons. Each reason is a grammatical sentence that names the real value. |
| 15 | Trained-on label | metrics.json records the patches and row counts. Footer and Insights show "Trained on patches 16.18–16.20, tested on 16.21 (24,683 games)". |
| 16 | Calibration chart | Insights page shows predicted-vs-actual bins for the state and combined models, using the same bins as the reality check. |
| 17 | Champion table | Search by name, sort by win rate / games / name, minimum-games filter (default 30). |
| 19 | One-command refresh | `python -m lolwp.refresh [--ids N --details N --timelines N]` runs collection (resumable), build, train and insights in order. It stops cleanly with the existing message if the key expired. |
| 20 | `--limit` as a total | `--limit N` means "stop once the store holds N of this item", so reruns don't overshoot. |
| 21 | Drop reasons | `build` counts drops by reason (wrong queue, early surrender, role missing, other platform, malformed) and prints the counts. They also go into metrics.json. |
| 22 | Reject booleans | State values must be real integers. `true`/`false` get a 422. |
| 23 | Unseen champion e2e | An API test predicts with a champion absent from the training stats and gets the 50% prior features and a valid response. |
| 24 | Layout check | New inputs and the result work at phone width (no horizontal scroll), in dark mode, and by keyboard alone (visible focus, labels on every input). Checked by reading the CSS and markup plus `npm run build`/lint. I can't open a browser here. |

## Left to the builder

- Exact API shape for unknown fields. The suggested shape is `state: dict[str, int | None]` plus `minute: 10 | 15`.
  The response gains:
  - `omitted: list[str]`;
  - per contribution `value` (the raw input) and `logit`;
  - `calibration: {lo, hi, win_rate, n}`;
  - `reliability: str`.
- The cache size limit and the format of the shipped training rows (Parquet in `artifacts/`, state features only,
  no match IDs needed).
- Bin edges for calibration (10 equal-width bins of predicted probability; empty bins dropped).
- Component split in the frontend.
