# League of Legends Win Predictor

**How much does the draft decide a ranked League game, compared with the first 15 minutes?**

> ⚠️ **Fill in after the real data run.** Every `TBD` below comes from `artifacts/metrics.json` and
> `artifacts/insights.json`. The pipeline, models, API, and web app are built and tested. The numbers
> don't exist until the Riot data is collected and the models are trained (see [Run it locally](#run-it-locally)).

## Headline findings

- Draft only: test AUC **TBD**, log loss **TBD** vs baseline **TBD**. *(Does the draft model beat the baseline? Say so either way.)*
- Game state at 15:00: test AUC **TBD**. Adding the draft: **TBD** (combined AUC **TBD**).
- At 15:00, one dragon vs one tower, gold held equal: **TBD** (p = **TBD**).
- Blue side wins **TBD**% of games (n = **TBD**, p = **TBD**).
- LightGBM vs logistic regression: **TBD** AUC difference. *(Above or below the 0.01 threshold?)*

![Screenshot of the Predict page](docs/screenshot.png) <!-- add after deploying -->

**Live app:** TBD (Render free tier: the first load can take ~30 s while it wakes up.)

## Data

- Source: official Riot Games API with my own developer key. Only public ranked match data.
- Population: ranked solo/duo (queue 420), SG2 server, Emerald and above. Players are seeded from the Emerald and Diamond ladders (divisions I–IV, pages 1–2) and Master, Grandmaster, and Challenger.
- Window: the last 42 days at collection time (about 3 patches). Target: ~20,000 matches with details, ~8,000 with 15-minute timelines.
- Collection is rate-limited (≥ 1.25 s between calls, honours `Retry-After`) and resumable. Progress lives in SQLite, so an expired dev key means "update `.env` and rerun".
- Cleaning: drop remakes, early surrenders, and any match where a player lacks a clear role. Game-state rows also need ≥ 16 minutes.
- Split by time: the **newest patch is the test set**. Older patches are training data.

## Method

| Model | Features | Used for |
|---|---|---|
| Baseline | champion win-rate difference → `sigmoid(10·x)`, no fitting | comparison |
| Draft | logistic regression on 15 draft features | Draft mode |
| Game state | logistic regression on 15 state features at 15:00 | Game-state mode |
| Combined | logistic regression on the draft log-odds + state features | Both mode |
| Benchmarks | LightGBM for each of the three | "is non-linear clearly better?" |

- **Draft features**: average champion win rate, the five lane matchups, bot-duo synergy, and class tags plus physical/magic damage from Data Dragon. All Blue minus Red. Win rates are shrunk toward 50% with a 20-game prior. Unseen champions get exactly 50%.
- **No target leakage**: training rows use 5-fold *out-of-fold* win rates, so a match's own result never describes it. The combined model's draft input is out-of-fold too. Test rows and the app use stats from the full training set. A test asserts that no test-patch match feeds the saved stats.
- **Exact explanations**: in log-odds each feature adds `coef × standardized value`. The app adds them largest-first and converts to probability after each step, so the waterfall bars sum exactly to the prediction.
- **Statistics**: a logistic regression of the win on the 15:00 state gives odds ratios with 95% intervals, plus a Wald test of "dragon = tower". This shows association, not causation.

**Rejected alternatives**: LightGBM as the main model (approximate SHAP explanations, small expected gain), neural draft models (need far more data, hard to explain), random forest (no better than LightGBM, no easier to explain), and player-history features (10 Riot IDs and many more API calls per prediction).

## Results (test patch TBD)

| Model | AUC | Log loss | Accuracy | Brier | CV AUC |
|---|---|---|---|---|---|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| Draft LR | TBD | TBD | TBD | TBD | TBD |
| Draft LGBM | TBD | TBD | TBD | TBD | TBD |
| State LR | TBD | TBD | TBD | TBD | TBD |
| State LGBM | TBD | TBD | TBD | TBD | TBD |
| Combined LR | TBD | TBD | TBD | TBD | TBD |
| Combined LGBM | TBD | TBD | TBD | TBD | TBD |

**Drift check**: each LR is also trained on all but the newest two patches and scored on the second-newest patch and on the newest. The AUC drop between them shows how much one patch of balance changes hurts. Result: TBD.

## Limitations

- One server (SG2) and one rank band (Emerald+). Results may not transfer.
- About 3 patches of data. Rare picks and matchups lean heavily on the 50% prior.
- No player skill, mastery, or live data.
- The objective statistics are associations. Teams that take dragons differ in ways the model doesn't see.

## Run it locally

Needs Python 3.11+ and Node 20+.

```bash
python -m venv .venv
.venv\Scripts\activate                # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # then put your key from developer.riotgames.com in RIOT_API_KEY

# 1. Collect (resumable; rerun any step after a key renewal)
python -m lolwp.riot.collect check --riot-id "Name#TAG"   # sanity check: prints a PUUID and SG2_ match IDs
python -m lolwp.riot.collect seed
python -m lolwp.riot.collect ids --limit 700
python -m lolwp.riot.collect details --limit 20000      # long; leave running
python -m lolwp.riot.collect timelines --limit 8000
python -m lolwp.riot.collect status

# 2. Champion data, features, models, statistics
python -c "from lolwp import champions; champions.download('data')" && cp data/ddragon/champions.json artifacts/
python -m lolwp.features.build        # want ≥ 1,500 draft rows in the newest (test) patch
python -m lolwp.models.train
python -m lolwp.models.insights

# 3. Tests (they also run the whole pipeline on synthetic data, so they pass before step 1)
pytest

# 4. App
cd web && npm ci && npm run build && cd ..
uvicorn lolwp.api.main:app --reload   # http://127.0.0.1:8000  (API docs at /docs)
# Frontend dev server with hot reload: cd web && npm run dev  (proxies /api to :8000)
```

The notebooks in `notebooks/` (EDA, models, statistics) read `data/` and `artifacts/`. Run them after step 2 and write the interpretation cells.

Deploy: `render.yaml` describes one Render web service. It builds the frontend, installs Python deps, and serves both from FastAPI.

---

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
