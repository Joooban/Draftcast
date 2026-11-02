# Draftcast

A League of Legends win predictor. **How much does the draft decide a ranked game, compared with the first 10 or 15 minutes?**

Pick a draft, or type what the scoreboard shows at 10:00 or 15:00, and Draftcast gives each team's win chance with a plain-language verdict ("Blue is favored: 68%, mostly because Blue leads by 2,300 gold"), the factors that moved it in percentage points, and an expandable full reasoning: an exact waterfall, every factor's input and log-odds, a reality check from test games and how far to trust the model.

## Headline findings

- **The draft barely decides the game.** Draft only: test AUC **0.529**, log loss **0.6923** vs baseline **0.6924** (AUC 0.524). The model edges past the no-fit baseline, but both are close to a coin flip (0.5).
- **The early game decides much more.** Game state at 15:00: test AUC **0.821** (74.2% accuracy); at 10:00 it is already **0.755** (68.7%). Adding the draft: **+0.002** (combined AUC **0.823**).
- **A dragon is worth more than a tower** at 15:00 with gold held equal: odds ratio 1.32 per dragon vs 0.92 per tower, a gap of 0.36 log-odds (p = 1×10⁻⁵⁰). Tower gold is already in the gold lead, so the tower term measures what a tower adds beyond its gold.
- **No blue-side advantage**: Blue wins **49.6%** of games (n = 24,683, p = 0.17).
- **LightGBM doesn't help**: it trails logistic regression by **0.004–0.015** AUC in every mode, so it never clears the +0.01 bar and the explainable model runs the app.

![Screenshot of the Predict page](docs/screenshot.png) <!-- add after deploying -->

**Live app:** https://draftcast.onrender.com (Render free tier: the first load can take ~30 s while it wakes up.)

## Data

- Source: official Riot Games API with my own developer key. Only public ranked match data.
- Population: ranked solo/duo (queue 420), SG2 server, Emerald and above. Players are seeded from the Emerald and Diamond ladders (divisions I–IV, pages 1–2) and Master, Grandmaster, and Challenger.
- Window: the last 42 days at collection time (about 3 patches). Collected: 25,454 matches with details, plus timelines for every game of 16+ minutes; 24,683 usable after cleaning (patches 16.18–16.21). Matches the same players played on other servers (OC1, TW2) are dropped.
- Collection is rate-limited (≥ 1.25 s between calls, honours `Retry-After`) and resumable. Progress lives in SQLite, so an expired dev key means "update `.env` and rerun".
- Cleaning: drop remakes, early surrenders, and any match where a player lacks a clear role (counts per reason are in `artifacts/build_summary.json`: 642 early surrenders, 98 other-server games, 31 unclear roles). Game-state rows also need ≥ 16 minutes, for both the 10:00 and 15:00 snapshots, so the two are the same games. Rift Herald is not a feature: it spawns after 15:00, so it was 0 in every game. XP is not a feature either: players can't see it on the scoreboard, and total level carries the same signal.
- Split by time: the **newest patch is the test set**. Older patches are training data.

## Method

| Model | Features | Used for |
|---|---|---|
| Baseline | champion win-rate difference → `sigmoid(10·x)`, no fitting | comparison |
| Draft | logistic regression on 15 draft features | Draft mode |
| Game state | logistic regression on 13 state features at 10:00 or 15:00 | Game-state mode |
| Combined | logistic regression on the draft log-odds + state features | Both mode |
| Benchmarks | LightGBM for each of the three | "is non-linear clearly better?" |

- **Draft features**: average champion win rate, the five lane matchups, bot-duo synergy, and class tags plus physical/magic damage from Data Dragon. All Blue minus Red. Win rates are shrunk toward 50% with a 20-game prior. Unseen champions get exactly 50%.
- **No target leakage**: training rows use 5-fold *out-of-fold* win rates, so a match's own result never describes it. The combined model's draft input is out-of-fold too. Test rows and the app use stats from the full training set. A test asserts that no test-patch match feeds the saved stats.
- **Exact explanations**: in log-odds each feature adds `coef × standardized value`. The app adds them largest-first and converts to probability after each step, so the waterfall bars sum exactly to the prediction.
- **Unknown fields**: anything left blank is unknown, never "even". The API fits a logistic regression on just the known fields from shipped training rows, scores it on the test patch, and caches it, so the accuracy and reality check shown are for the model that actually predicted.
- **Reality check**: test-patch games are binned by predicted probability; each prediction shows how often Blue actually won in its bin. The Insights page plots all bins.
- **Statistics**: a logistic regression of the win on the 15:00 state gives odds ratios with 95% intervals, plus a Wald test of "dragon = tower". This shows association, not causation.

**Rejected alternatives**: LightGBM as the main model (approximate SHAP explanations, small expected gain), neural draft models (need far more data, hard to explain), random forest (no better than LightGBM, no easier to explain), and player-history features (10 Riot IDs and many more API calls per prediction).

## Results (test patch 16.21)

Train: 17,838 draft / 17,087 state rows. Test: 6,845 / 6,522.

| Model | AUC | Log loss | Accuracy | Brier | CV AUC |
|---|---|---|---|---|---|
| Baseline | 0.523 | 0.6924 | 51.6% | 0.2496 | 0.542 ± 0.005 |
| Draft LR | 0.529 | 0.6923 | 51.7% | 0.2496 | 0.543 ± 0.006 |
| Draft LGBM | 0.514 | 0.6996 | 50.7% | 0.2531 | 0.529 ± 0.009 |
| State LR | 0.821 | 0.5180 | 74.2% | 0.1727 | 0.821 ± 0.003 |
| State LGBM | 0.816 | 0.5264 | 73.8% | 0.1754 | 0.815 ± 0.003 |
| Combined LR | 0.823 | 0.5153 | 74.5% | 0.1718 | 0.823 ± 0.004 |
| Combined LGBM | 0.819 | 0.5219 | 73.9% | 0.1739 | 0.817 ± 0.003 |
| State LR, 10:00 | 0.755 | 0.5871 | 68.7% | 0.2011 | 0.746 ± 0.006 |
| Combined LR, 10:00 | 0.758 | 0.5845 | 69.0% | 0.2000 | 0.751 ± 0.007 |

**Drift check**: each LR is also trained on all but the newest two patches and scored on the second-newest patch and on the newest. The AUC drop between them shows how much one patch of balance changes hurts. Result: the draft model falls from 0.534 (16.20) to 0.515 (16.21), so draft win rates go stale within a patch. Game state barely moves (0.825 → 0.820), and neither does the combined model (0.825 → 0.821).

## Limitations

- One server (SG2) and one rank band (Emerald+). Results may not transfer.
- About 3 patches of data. Rare picks and matchups lean heavily on the 50% prior.
- No player skill, mastery, or live data.
- The objective statistics are associations. Teams that take dragons differ in ways the model doesn't see.

## Run it locally

Needs Python 3.11+ and Node 20+.

```bash
git clone https://github.com/Joooban/Draftcast.git && cd Draftcast
python -m venv .venv
.venv\Scripts\activate                # Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt   # everything: collect, train, notebooks, tests (the app alone needs requirements.txt)
cp .env.example .env                  # then put your key from developer.riotgames.com in RIOT_API_KEY

# 1. Collect (resumable; rerun any step after a key renewal)
python -m lolwp.riot.collect check --riot-id "Name#TAG"   # sanity check: prints a PUUID and SG2_ match IDs
python -m lolwp.riot.collect seed
python -m lolwp.riot.collect ids --limit 20000          # --limit is a total: stop once the store holds this many
python -m lolwp.riot.collect details --limit 20000      # long; leave running
python -m lolwp.riot.collect timelines --limit 8000
python -m lolwp.riot.collect status

# Or all of it in one go once seeded (targets are store totals; stops cleanly if the key expires):
python -m lolwp.refresh --ids 20000 --details 20000 --timelines 8000

# 2. Champion data, features, models, statistics
python -c "from lolwp import champions; champions.download('data')" && cp data/ddragon/champions.json artifacts/
python -m lolwp.features.build        # want ≥ 1,500 draft rows in the newest (test) patch
python -m lolwp.models.train
python -m lolwp.models.insights

# 3. Tests (they also run the whole pipeline on synthetic data, so they pass before step 1)
pytest
cd web && npm run check:live && cd ..   # live-entry summing rules

# 4. App
cd web && npm ci && npm run build && cd ..
uvicorn lolwp.api.main:app --reload   # http://127.0.0.1:8000  (API docs at /docs)
# Frontend dev server with hot reload: cd web && npm run dev  (proxies /api to :8000)
```

The notebooks in `notebooks/` (EDA, models, statistics) read `data/` and `artifacts/`. Run them after step 2. The interpretation cells hold drafts to rewrite in your own words.

Deploy: `render.yaml` describes one Render web service. It builds the frontend, installs Python deps, and serves both from FastAPI.

---

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
