# League of Legends Win Predictor: Design

**Date:** 2026-10-30
**Status:** Draft for review
**Owner:** Jhouvann Morden
**Sub-project:** 1 of 2 (League of Legends). Dota 2 is sub-project 2 and gets its own design later, reusing this one where possible.

---

## 1. Purpose

A data science portfolio project for a data science role. It must show the full workflow: data collection, cleaning, feature engineering, modeling, evaluation, statistics, and communicating results through an interactive web app.

**Audience:** hiring managers and reviewers, some of whom don't play League.

**Angle (justification):** Well-known League prediction work and datasets are several years old. The early game has changed a lot since then (voidgrubs, Atakhan, objective and tower reward changes). This project asks what predicts wins in the *current* game, and how much the draft matters compared with the first 15 minutes.

**Constraint:** Finish fast while learning. Prefer simple, explainable choices over complex ones.

## 2. Success criteria

1. A deployed web app with three prediction modes (Draft, Game state, Both), each showing a probability and an explanation of why.
2. Every model is evaluated on the newest patch, held out by time, and compared against a simple baseline using log loss, ROC-AUC, accuracy, and Brier score.
3. The draft model beats the baseline on test log loss. If it doesn't, that result is reported as a finding, not hidden.
4. An Insights page and README present the headline findings with charts, including draft-only vs. game-state vs. combined accuracy.
5. Automated tests for features, leakage, model sanity, and the API all pass.

## 3. Scope

**In:** ranked solo/duo (queue 420), SG2 server, Emerald and above, the most recent 3 patches at collection time, a draft model, a game-state model at 15:00, a combined model, the web app, and the README.

**Out:** Dota 2 (sub-project 2), player-history features (each player's win rate or mastery), neural networks, live in-game data, other servers or ranks, and pick/ban recommendations.

## 4. Data

**Source:** Riot API with the owner's own developer key. No one else's account or login is used. Data is pulled from other players' public ranked matches.

| Step | Endpoint | Purpose |
|---|---|---|
| Seed players | `league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}` on `sg2` | PUUIDs for Emerald, Diamond, Master+ players |
| Match IDs | `match/v5/matches/by-puuid/{puuid}/ids?queue=420` on `sea` | Recent ranked match IDs |
| Match details | `match/v5/matches/{id}` | Picks, roles, winner, patch, duration |
| Timeline | `match/v5/matches/{id}/timeline` | Per-minute state for game-state features |

**Volume:** about 20,000 matches with details, and about 8,000 of those with timelines (about 28,000 calls, roughly 10 hours on a dev key, over 2 runs).

**Rate limiting:** at most 1 request per 1.25 seconds. On 429, wait for `Retry-After`. Retry 5xx errors with backoff. Stop with a clear message on 401/403 (expired key). Collection is resumable: progress flags live in the database.

**Storage:** SQLite (`data/lol.db`) holds raw JSON, zlib-compressed, with timelines trimmed to minutes 0 to 15. Feature tables are written as Parquet files.

**Filters:** drop remakes and early surrenders, and games under 16 minutes for the game-state model only. Keep each match's patch.

**Split:** the newest patch is the test set and older patches are the training set. Tuning uses 5-fold cross-validation inside the training set only.

## 5. Features

All features are Blue minus Red unless noted.

**Draft features (computed from training matches only):**
- Team champion strength: difference in average smoothed champion win rate.
- Lane matchups: for each of Top, Jungle, Mid, Bot, and Support, the smoothed win rate of Blue's champion against Red's champion in that role (5 features).
- Bot duo synergy: difference in smoothed ADC+Support pair win rate.
- Team composition from Data Dragon: differences in tag counts (Tank, Fighter, Mage, Assassin, Marksman, Support) and in total physical vs. magic damage rating.
- Smoothing: empirical Bayes shrinkage toward 50% with a prior weight of 20 games. Champions, matchups, or pairs with no training data get 50%.
- Out-of-fold encoding: a training match's own result must not be counted in the stats used to describe it. Draft features for training rows are therefore computed with 5-fold out-of-fold stats (each fold uses stats built from the other 4). Test rows and the live API use stats built from the full training set. The combined model's draft input for training rows uses out-of-fold draft-model predictions for the same reason.

**Game-state features (at 15:00):** gold, XP, CS, level, kills, dragons, voidgrubs, heralds, towers, plates, wards placed, wards killed, plus first blood, first dragon, and first tower (encoded 1 for Blue, −1 for Red, 0 for none).

**Combined:** the draft model's log-odds output plus the game-state features.

**Never used as features:** game duration and any end-of-game statistic.

## 6. Models

| Model | Method | Role |
|---|---|---|
| Baseline | Difference in average champion win rate, mapped to a probability | Comparison point |
| Draft | Logistic regression on draft features | Draft mode |
| Game state | Logistic regression on game-state features | Game-state mode |
| Combined | Logistic regression on draft log-odds plus game-state features | Both mode |
| Benchmarks | LightGBM for each of the three | Checks whether a non-linear model is clearly better |

**Why logistic regression:** each feature's contribution is exact and adds up (coefficient × value in log-odds), so the app's "why" chart is precise and easy to read. It also fits a moderate dataset well.

**Rejected alternatives:**
- LightGBM as the main model: explanations are approximate (SHAP), and any accuracy gain is expected to be small. It's kept as a benchmark. If it beats logistic regression by more than 0.01 AUC on test, that's reported, but the explainable model still runs the app.
- Neural networks (champion embeddings, transformer draft models): need far more data, take longer to learn, and are hard to explain. Listed as future work.
- Random forest: usually worse than LightGBM and no easier to explain.
- Player-history features: would need 10 Riot IDs per prediction and many more API calls. Out of scope.

**Statistics component:** a logistic regression on the game-state data, reporting odds ratios with 95% confidence intervals, plus a formal test of whether one dragon is worth the same as one tower with gold held equal. The write-up notes that this shows association, not causation.

## 7. Explanations

- Each prediction returns a baseline (the model's prediction for an average game: the sigmoid of the intercept on standardized features, which is close to the training-set Blue win rate) and a list of contributions, one per feature, in waterfall order. Contributions are added in order of largest absolute log-odds effect, and each bar is the change in probability when that feature is added, so the bars sum exactly to the final probability minus the baseline.
- Both mode split: the draft share is the probability change from adding the draft input to the baseline, and the game-state share is the rest.
- The top 3 contributions are turned into plain-English sentences from templates (for example, "Blue's mid lane matchup historically wins 54% of games: +3%").
- Both mode also returns how much of the result came from the draft and how much from the game state.

## 8. Application

**Stack:** React + TypeScript + Vite frontend, with Plotly.js for charts, served as static files by a FastAPI backend. Deployed as one Render web service.

**Pages:**
1. **Predict:** a mode switch (Draft / Game state / Both). Draft input has 5 role slots per team, with searchable champion selection and Data Dragon icons. Game-state input uses sliders. Output shows the win probability, a waterfall "why" chart, the top 3 reasons, and in Both mode, the draft vs. game-state split.
2. **Insights:** champion win rates with confidence intervals, win rate vs. gold lead at 15:00, objective odds ratios, and the model accuracy comparison.
3. **Method:** data, method choices, rejected alternatives, limitations, and the Riot legal disclaimer.

**API:**
- `GET /api/champions` returns `[{id, name, icon_url, tags}]`
- `POST /api/predict` with body `{mode: "draft"|"state"|"both", blue?: {top, jungle, mid, bot, support}, red?: {...}, state?: {...}}` returns `{probability, baseline, contributions: [{feature, label, value_pct}], reasons: [str], split?: {draft_pct, state_pct}}`
- `GET /api/insights` returns precomputed chart data

**Validation (returns HTTP 422 with a readable message, shown in the UI):**
- Draft and Both modes need all 10 picks.
- No champion can appear twice.
- Unknown champion IDs are rejected.
- Game-state values must be within the slider ranges.
- A mode that's missing its required input is rejected.

## 9. Testing

- **Features:** a small saved sample match with known expected values (for example, gold difference at minute 15).
- **Leakage:** champion, matchup, and synergy stats are built only from training match IDs. A test asserts that no test-set match ID contributes.
- **Model sanity:** all probabilities are within [0, 1]. Swapping Blue and Red gives a mean |p + p_swapped − 1| ≤ 0.05 on the test set.
- **API:** valid requests for each mode, plus every validation rule in section 8.

## 10. Deliverables

1. A GitHub repository (key, raw data, and `.env` excluded) containing the collector, pipeline, notebooks, API, frontend, and tests.
2. The deployed Render app.
3. A README that starts with the findings: question, headline answer, key chart, data, method, results, limitations, and how to run it.
4. A new project entry on the portfolio site.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Draft model barely beats a coin flip | Expected. Framed as the headline finding about how much the draft matters. |
| Dev key expires every 24 hours | Resumable collector. Apply for a Personal API Key in parallel. |
| API response fields change | Collector checks for required fields and logs skipped matches with a reason. |
| New patch arrives mid-collection | Patch is stored per match, and the test set is defined as the newest patch present. |
| Render free tier sleeps | Accepted. The README notes a possible delay on first load. |
