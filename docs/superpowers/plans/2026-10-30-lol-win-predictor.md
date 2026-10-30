# League of Legends Win Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect recent SG2 ranked matches, train explainable draft, game-state, and combined win models, and serve them through a FastAPI + React web app with "why" charts.

**Architecture:** A Python package (`lolwp`) handles collection (Riot API → SQLite), feature building (→ Parquet), training (→ joblib + JSON artifacts), and a FastAPI server that loads the artifacts. A React + TypeScript + Vite frontend is built into static files and served by the same FastAPI app, deployed as one Render service.

**Tech Stack:** Python 3.11+, requests, pandas, pyarrow, scikit-learn, lightgbm, statsmodels, joblib, FastAPI, uvicorn, pydantic v2, pytest; Node 20+, React 18, TypeScript, Vite, react-router-dom, plotly.js-dist-min, react-plotly.js.

**Spec:** `docs/superpowers/specs/2026-10-30-lol-win-predictor-design.md`

## Global Constraints

- Queue 420 (ranked solo/duo) only. Platform `sg2`, regional routing `sea`, account routing `asia`.
- Seed tiers: EMERALD I–IV, DIAMOND I–IV (pages 1–2), MASTER/GRANDMASTER/CHALLENGER division I (page 1).
- Match window: `startTime` = now − 42 days (about 3 patches). Store patch as `major.minor` from `gameVersion`.
- Targets: about 20,000 matches with details, about 8,000 with timelines.
- Rate limit: at least 1.25 s between requests. 429 → sleep `Retry-After` (default 10). 5xx → retry with backoff, max 5 attempts. 401/403 → raise `KeyExpiredError`. 404 → return `None`.
- Raw JSON stored zlib-compressed. Timelines trimmed to frames 0–15.
- Valid match: queue 420, not an early surrender or remake, all 10 participants have a `teamPosition` in {TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY}, one of each per team. Game-state rows also require `gameDuration` ≥ 960 s and at least 16 frames.
- Roles in the app and API use: `top`, `jungle`, `mid`, `bot`, `support` (mapping TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY).
- Test set = matches from the newest patch present. Train = all older patches.
- Smoothing prior weight = 20 games at 0.5. Out-of-fold K = 5, seed 42.
- `game_duration` and end-of-game stats are never model features.
- LightGBM params: `n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1`.
- Symmetry tolerance: mean |p + p_swapped − 1| ≤ 0.05.
- Secrets: `RIOT_API_KEY` only in `.env`, never committed. `data/` is git-ignored. `artifacts/` is committed.
- Every page footer shows: "This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are trademarks or registered trademarks of Riot Games, Inc."

## Review Focus

1. **A champion released after training data (or never seen in training)** must get 0.5 for every stat and still predict. Test owned by Task 8.
2. **Dev key expires mid-collection.** Rerunning must resume without re-fetching stored matches or creating duplicates. Test owned by Task 4.
3. **A match where `teamPosition` is blank for someone** (common in older/odd games) must be excluded, not crash parsing. Test owned by Task 6.
4. **Both mode with game state filled but a partial draft** must return 422, not a draft-less prediction. Test owned by Task 15.
5. **Slider values outside the allowed range sent straight to the API** must return 422. Test owned by Task 15.

---

## File Structure

```
lol-win-predictor/
├── .env.example                  # RIOT_API_KEY=, PLATFORM=sg2, REGION=sea
├── .gitignore                    # .env, .venv/, data/, __pycache__/, web/node_modules/, web/dist/
├── requirements.txt
├── pytest.ini                    # testpaths = tests
├── README.md
├── docs/superpowers/{specs,plans}/
├── lolwp/
│   ├── __init__.py
│   ├── config.py                 # settings, constants (roles, ranges, paths)
│   ├── riot/
│   │   ├── __init__.py
│   │   ├── client.py             # RiotClient: rate-limited GET
│   │   ├── store.py              # SQLite storage
│   │   └── collect.py            # CLI: seed / ids / details / timelines
│   ├── champions.py              # Data Dragon download + lookup
│   ├── features/
│   │   ├── __init__.py
│   │   ├── parse.py              # match JSON → MatchRecord
│   │   ├── state.py              # timeline → game-state features
│   │   ├── draft_stats.py        # smoothed win rates, OOF
│   │   ├── draft.py              # picks → draft features
│   │   └── build.py              # CLI: write Parquet datasets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── explain.py            # waterfall contributions
│   │   ├── train.py              # CLI: train + evaluate + save
│   │   └── insights.py           # CLI: stats model + insights.json
│   └── api/
│       ├── __init__.py
│       ├── schemas.py            # pydantic request/response
│       ├── predictor.py          # loads artifacts, predicts
│       └── main.py               # FastAPI app + static files
├── artifacts/                    # committed: models + JSON
├── data/                         # ignored: lol.db, ddragon/, *.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_models.ipynb
│   └── 03_stats.ipynb
├── tests/
│   ├── fixtures/                 # small handcrafted JSON
│   └── test_*.py
└── web/                          # Vite React TS app
    └── src/
        ├── api.ts
        ├── types.ts
        ├── App.tsx
        ├── components/{ChampionPicker,DraftInput,StateInput,ResultPanel,WaterfallChart,Footer}.tsx
        └── pages/{Predict,Insights,Method}.tsx
```

---

### Task 1: Project scaffold and config

**Concept:** a reproducible environment and one place for settings, so nothing is hard-coded in five files.

**Files:**
- Create: `requirements.txt`, `.gitignore`, `.env.example`, `.env`, `pytest.ini`, `lolwp/__init__.py`, `lolwp/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `lolwp.config` with
  - `ROLES: list[str] = ["top","jungle","mid","bot","support"]`
  - `RIOT_POSITION_TO_ROLE: dict[str,str] = {"TOP":"top","JUNGLE":"jungle","MIDDLE":"mid","BOTTOM":"bot","UTILITY":"support"}`
  - `STATE_FEATURES: list[str]` (order: gold_diff, xp_diff, cs_diff, level_diff, kills_diff, dragons_diff, grubs_diff, heralds_diff, towers_diff, plates_diff, wards_placed_diff, wards_killed_diff, first_blood_blue, first_dragon_blue, first_tower_blue)
  - `STATE_RANGES: dict[str, tuple[int,int]]` with gold_diff (−15000, 15000), xp_diff (−15000, 15000), cs_diff (−200, 200), level_diff (−20, 20), kills_diff (−30, 30), dragons_diff (−3, 3), grubs_diff (−6, 6), heralds_diff (−1, 1), towers_diff (−6, 6), plates_diff (−15, 15), wards_placed_diff (−60, 60), wards_killed_diff (−30, 30), first_* (−1, 1)
  - `PRIOR_GAMES = 20`, `OOF_FOLDS = 5`, `SEED = 42`, `MIN_INTERVAL_S = 1.25`
  - `Settings` dataclass: `api_key, platform, region, db_path, data_dir, artifacts_dir`; `load_settings() -> Settings` reading `.env`

- [ ] **Step 1:** Create the folder, venv, and install:

```
mkdir lol-win-predictor && cd lol-win-predictor && git init
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install requests python-dotenv pandas pyarrow numpy scikit-learn lightgbm statsmodels joblib fastapi "uvicorn[standard]" pydantic httpx pytest jupyter matplotlib seaborn
pip freeze > requirements.txt
```

- [ ] **Step 2:** Write `.gitignore`, `.env.example`, `.env` (your real key), `pytest.ini` per the file structure. Copy the spec into `docs/superpowers/specs/` and this plan into `docs/superpowers/plans/`.

- [ ] **Step 3: Write the failing test**

```python
from lolwp import config

def test_roles_and_mapping_align():
    assert set(config.RIOT_POSITION_TO_ROLE.values()) == set(config.ROLES)

def test_every_state_feature_has_range():
    assert set(config.STATE_FEATURES) == set(config.STATE_RANGES)
    assert len(config.STATE_FEATURES) == 15

def test_load_settings_defaults(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
    monkeypatch.delenv("PLATFORM", raising=False)
    s = config.load_settings()
    assert s.api_key == "RGAPI-test" and s.platform == "sg2" and s.region == "sea"
```

- [ ] **Step 4:** Run `pytest tests/test_config.py -v`. Expected: FAIL (module not found).
- [ ] **Step 5:** Implement `lolwp/config.py` with the values above.
- [ ] **Step 6:** Run `pytest -v`. Expected: 3 passed.
- [ ] **Step 7:** Commit: `git add . && git commit -m "chore: scaffold project and config"`

---

### Task 2: Rate-limited Riot client

**Concept:** APIs have limits; a polite client is part of real data engineering.

**Files:**
- Create: `lolwp/riot/__init__.py`, `lolwp/riot/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `Settings`, `MIN_INTERVAL_S`
- Produces:
  - `class KeyExpiredError(Exception)`
  - `class RiotClient(api_key: str, session=None, sleep=time.sleep, clock=time.monotonic, min_interval: float = 1.25)`
  - `RiotClient.get(url: str, params: dict | None = None) -> dict | list | None`

- [ ] **Step 1: Write the failing tests** using a fake session (class with `.get()` returning objects with `status_code`, `headers`, `json()`), a fake `sleep` that records calls, and a fake clock.

```python
def test_returns_json_on_200(): ...            # get() == {"ok": 1}
def test_404_returns_none(): ...
def test_403_raises_key_expired(): ...         # pytest.raises(KeyExpiredError)
def test_429_sleeps_retry_after_then_succeeds():
    # responses: 429 (Retry-After: 7), then 200
    # assert 7 in recorded_sleeps and result == {"ok": 1}
def test_500_retries_then_gives_up_after_5(): ...   # 5x 500 -> returns None, session called 5 times
def test_spacing_between_calls():
    # clock frozen at 0.0; two get() calls -> a sleep of ~1.25 recorded before the second
def test_sends_token_header(): ...             # session headers contain X-Riot-Token
```

- [ ] **Step 2:** Run `pytest tests/test_client.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `RiotClient` per Global Constraints (5xx backoff: sleep `5 * attempt`).
- [ ] **Step 4:** Run tests. Expected: 7 passed.
- [ ] **Step 5:** Commit: `feat: rate-limited Riot API client`

---

### Task 3: SQLite store

**Concept:** store raw data once, so you never re-download when features change.

**Files:**
- Create: `lolwp/riot/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces `class Store(db_path: str)` with:
  - `add_players(rows: list[tuple[str, str, str]]) -> int` (puuid, tier, division; ignores duplicates; returns new count)
  - `players_todo(limit: int) -> list[str]` / `mark_player_done(puuid: str) -> None`
  - `add_match_ids(ids: list[str]) -> int`
  - `details_todo(limit: int) -> list[str]`
  - `save_details(match_id: str, match: dict | None) -> None` (None marks fetched with no data)
  - `timelines_todo(limit: int, seed: int) -> list[str]` (random sample of stored valid-looking details: duration ≥ 960, no timeline yet)
  - `save_timeline(match_id: str, timeline: dict | None) -> None` (trims to frames[:16])
  - `iter_matches() -> Iterator[tuple[str, dict, dict | None]]` (match_id, details, timeline-or-None; decompressed)
  - `counts() -> dict[str, int]` (players, match_ids, details, timelines)
- Tables: `players(puuid PK, tier, division, done)`, `match_ids(match_id PK, details_fetched, timeline_fetched)`, `matches(match_id PK, details BLOB, timeline BLOB, duration INT)`

- [ ] **Step 1: Write the failing tests** (use `tmp_path`):

```python
def test_add_players_ignores_duplicates(tmp_path): # add same puuid twice -> counts()["players"] == 1
def test_details_roundtrip(tmp_path):              # save_details then iter_matches returns same dict, timeline None
def test_details_todo_excludes_fetched(tmp_path):  # after save_details(None), id not in details_todo
def test_timeline_trimmed_to_16_frames(tmp_path):  # timeline with 30 frames -> stored has 16
def test_timelines_todo_skips_short_games(tmp_path): # duration 900 not returned, 1200 returned
def test_reopen_keeps_progress(tmp_path):          # new Store on same path -> counts unchanged
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement with `sqlite3` + `zlib` + `json`. **Step 4:** Run, expect 6 passed.
- [ ] **Step 5:** Commit: `feat: SQLite store for raw matches`

---

### Task 4: Collector CLI and first real run

**Concept:** seeding from ranked ladders gives a sample of the population, not one player's history.

**Files:**
- Create: `lolwp/riot/collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `RiotClient.get`, `Store`, `Settings`
- Produces CLI `python -m lolwp.riot.collect {check,seed,ids,details,timelines,status} [--limit N] [--riot-id NAME#TAG]` and functions:
  - `check(client, settings, riot_id: str) -> None` (prints PUUID and 5 match IDs; riot_id like `Name#TAG`, URL-quoted)
  - `seed(client, store, settings) -> int`
  - `collect_ids(client, store, settings, limit: int, per_player: int = 50, now: float | None = None) -> int` (uses `startTime = now - 42*86400`, `queue=420`, `count=per_player`)
  - `collect_details(client, store, settings, limit: int) -> int`
  - `collect_timelines(client, store, settings, limit: int) -> int`
  - Each loop catches `KeyExpiredError`, prints "Key expired — update .env and rerun", and returns cleanly.

- [ ] **Step 1: Write the failing tests** with a fake client that maps URLs to responses:

```python
def test_seed_stores_puuids_from_entries(tmp_path)
def test_collect_ids_passes_start_time_and_queue(tmp_path)   # inspect recorded params
def test_collect_details_resume_after_key_expiry(tmp_path):
    # fake client raises KeyExpiredError on the 3rd details call
    # first run stores 2; second run (healthy client) stores the rest; no id fetched twice
def test_seed_skips_entries_without_puuid(tmp_path)
```

- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect 4 passed.
- [ ] **Step 5:** Get a dev key from developer.riotgames.com, put it in `.env`. Run `python -m lolwp.riot.collect check --riot-id "YourName#TAG"`. Expected: a PUUID and match IDs starting `SG2_`. (Any Riot ID works, including your friend's public Riot ID. No login needed.)
- [ ] **Step 6:** Run, in order: `seed`, then `ids --limit 700`, then start `details --limit 20000` (leave running overnight; rerun after each key renewal). Then `timelines --limit 8000`. Check progress with `python -m lolwp.riot.collect status` (prints `counts()`).
- [ ] **Step 7:** Commit: `feat: resumable match collector`

While collection runs, continue with Tasks 5–9 (they use fixtures, not real data).

---

### Task 5: Champion data from Data Dragon

**Files:**
- Create: `lolwp/champions.py`, `tests/fixtures/ddragon_champions.json` (3 champions: Ahri key 103 tags [Mage, Assassin] attack 3 magic 8; Garen key 86 tags [Fighter, Tank] attack 7 magic 1; Jinx key 222 tags [Marksman] attack 9 magic 2)
- Test: `tests/test_champions.py`

**Interfaces:**
- Produces:
  - `TAGS = ["Tank","Fighter","Mage","Assassin","Marksman","Support"]`
  - `@dataclass Champion(id: int, key: str, name: str, tags: list[str], attack: int, magic: int, icon_url: str)`
  - `download(data_dir: str) -> str` (fetches `versions.json`, then `cdn/{v}/data/en_US/champion.json`; saves to `data/ddragon/champions.json`; returns version)
  - `load(path: str) -> dict[int, Champion]` (icon `https://ddragon.leagueoflegends.com/cdn/{v}/img/champion/{key}.png`)

- [ ] **Step 1: Failing tests:** `load` on the fixture returns 3 champions; `load(...)[103].tags == ["Mage","Assassin"]`; icon URL contains the version and `Ahri.png`.
- [ ] **Step 2–4:** Run (FAIL), implement, run (3 passed).
- [ ] **Step 5:** Run `python -c "from lolwp import champions; print(champions.download('data'))"`. Expected: a version string. Copy `data/ddragon/champions.json` to `artifacts/champions.json` (the API uses the artifact copy).
- [ ] **Step 6:** Commit: `feat: Data Dragon champion data`

---

### Task 6: Parse matches

**Concept:** turn messy API JSON into one clean record per match, with explicit reasons for dropping bad rows.

**Files:**
- Create: `lolwp/features/__init__.py`, `lolwp/features/parse.py`, `tests/fixtures/match_valid.json` (10 participants with participantId 1–10, teamId 100 for 1–5 and 200 for 6–10, positions TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY each side, championIds; teams win: 100 true; queueId 420; gameDuration 1800; gameVersion "15.21.716.1234")
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces:
  - `@dataclass MatchRecord(match_id: str, patch: str, blue_win: int, duration_s: int, blue: dict[str,int], red: dict[str,int])` (role → championId)
  - `parse_match(match: dict) -> MatchRecord | None` (None = invalid per Global Constraints)
  - `patch_key(patch: str) -> tuple[int,int]` (for sorting "15.9" < "15.21")

- [ ] **Step 1: Failing tests:**

```python
def test_valid_match_parsed():  # patch == "15.21", blue_win == 1, blue["mid"] == fixture's mid champ
def test_wrong_queue_rejected():          # queueId 440 -> None
def test_early_surrender_rejected():      # any participant gameEndedInEarlySurrender True -> None
def test_blank_team_position_rejected():  # one teamPosition "" -> None   (Review Focus 3)
def test_duplicate_role_rejected():       # two TOPs on blue -> None
def test_patch_key_orders_numerically():  # sorted(["15.9","15.21","15.10"], key=patch_key) == ["15.9","15.10","15.21"]
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (6 passed).
- [ ] **Step 5:** Commit: `feat: parse match details`

---

### Task 7: Game-state features

**Files:**
- Create: `lolwp/features/state.py`, `tests/fixtures/timeline_small.json`
- Test: `tests/test_state.py`

**Fixture (handcrafted, 16 frames, 60000 ms apart):** frame 15 participantFrames: blue players each `totalGold` 5000, `xp` 6000, `minionsKilled` 100, `jungleMinionsKilled` 10, `level` 9; red players each 4800 / 5800 / 95 / 10 / 9. Events: frame 3 CHAMPION_KILL victimId 7 (t=150000); frame 6 ELITE_MONSTER_KILL killerTeamId 100 HORDE; frame 8 ELITE_MONSTER_KILL killerTeamId 200 DRAGON; frame 12 BUILDING_KILL TOWER_BUILDING teamId 200; frame 12 TURRET_PLATE_DESTROYED teamId 200 (×2); frame 14 WARD_PLACED creatorId 1 and 6 and 6; frame 15 CHAMPION_KILL victimId 2 at t=905000 (after 15:00, must be ignored).

**Interfaces:**
- Consumes: `STATE_FEATURES`
- Produces: `state_features(match: dict, timeline: dict, minute: int = 15) -> dict[str, int] | None` (keys exactly `STATE_FEATURES`; None if fewer than minute+1 frames). Team of a participant comes from match `participants[].teamId`, not from the ID number.

- [ ] **Step 1: Failing tests:**

```python
def test_diffs_at_15(fixtures):
    f = state_features(match, timeline)
    assert f["gold_diff"] == 1000 and f["xp_diff"] == 1000 and f["cs_diff"] == 25
    assert f["kills_diff"] == 1            # late kill ignored
    assert f["grubs_diff"] == 1 and f["dragons_diff"] == -1
    assert f["towers_diff"] == 1 and f["plates_diff"] == 2
    assert f["wards_placed_diff"] == -1
    assert f["first_blood_blue"] == 1 and f["first_dragon_blue"] == -1 and f["first_tower_blue"] == 1
    assert f["heralds_diff"] == 0 and f["wards_killed_diff"] == 0
def test_short_timeline_returns_none():  # frames[:10] -> None
def test_keys_match_config():            # list(f) == STATE_FEATURES
```

- [ ] **Step 2–4:** Run (FAIL), implement (event rules as in the spec: kills credited by victim's opposite team; BUILDING_KILL/TURRET_PLATE `teamId` = team that lost it), run (3 passed).
- [ ] **Step 5:** Commit: `feat: game-state features at 15 minutes`

---

### Task 8: Draft stats with smoothing and out-of-fold encoding

**Concept:** target encoding. If a match's own result is counted in its champion's win rate, the model cheats. Out-of-fold fixes that.

**Files:**
- Create: `lolwp/features/draft_stats.py`
- Test: `tests/test_draft_stats.py`

**Interfaces:**
- Consumes: `MatchRecord`, `PRIOR_GAMES`, `OOF_FOLDS`, `SEED`
- Produces:
  - `smooth(wins: float, games: float, prior: int = 20) -> float` = `(wins + prior*0.5) / (games + prior)`
  - `class DraftStats` with:
    - `@classmethod fit(records: list[MatchRecord]) -> DraftStats`
    - `champ_rate(champ: int) -> float`
    - `lane_rate(role: str, champ: int, enemy: int) -> float` (from `champ`'s side)
    - `duo_rate(bot: int, support: int) -> float`
    - `match_ids: set[str]` (which matches built these stats)
    - `to_json(path: str) -> None` / `@classmethod from_json(path: str) -> DraftStats`
  - `oof_stats(records: list[MatchRecord], k: int = 5, seed: int = 42) -> dict[str, DraftStats]` (match_id → stats fitted without that match's fold)

- [ ] **Step 1: Failing tests:**

```python
def test_smooth_values():
    assert smooth(0, 0) == 0.5 and smooth(3, 3) == pytest.approx(13/23)
def test_unseen_champion_gets_half():           # Review Focus 1
    s = DraftStats.fit(records)
    assert s.champ_rate(99999) == 0.5 and s.lane_rate("mid", 99999, 103) == 0.5 and s.duo_rate(1, 99999) == 0.5
def test_lane_rate_is_symmetric():
    assert s.lane_rate("mid", a, b) == pytest.approx(1 - s.lane_rate("mid", b, a))
def test_oof_excludes_own_match():
    oof = oof_stats(records)
    for mid, st in oof.items():
        assert mid not in st.match_ids
def test_json_roundtrip(tmp_path): ...
```

Build `records` in the test from 10 small synthetic `MatchRecord`s.

- [ ] **Step 2–4:** Run (FAIL), implement (lane symmetry: store each lane game from both sides), run (5 passed).
- [ ] **Step 5:** Commit: `feat: smoothed draft stats with out-of-fold encoding`

---

### Task 9: Draft features

**Files:**
- Create: `lolwp/features/draft.py`
- Test: `tests/test_draft.py`

**Interfaces:**
- Consumes: `DraftStats`, `Champion`, `TAGS`, `ROLES`
- Produces:
  - `DRAFT_FEATURES: list[str]` = `["champ_wr_diff", "lane_top", "lane_jungle", "lane_mid", "lane_bot", "lane_support", "duo_diff", "tag_Tank_diff", "tag_Fighter_diff", "tag_Mage_diff", "tag_Assassin_diff", "tag_Marksman_diff", "tag_Support_diff", "attack_diff", "magic_diff"]`
  - `draft_features(blue: dict[str,int], red: dict[str,int], stats: DraftStats, champs: dict[int, Champion]) -> dict[str, float]`
  - Definitions: `champ_wr_diff` = mean blue champ_rate − mean red; `lane_<role>` = `lane_rate(role, blue[role], red[role]) - 0.5`; `duo_diff` = blue duo_rate − red duo_rate; tag counts and attack/magic sums are blue − red. Unknown champion IDs in `champs` count as no tags, attack 0, magic 0.

- [ ] **Step 1: Failing tests:**

```python
def test_keys_and_order():          # list(f) == DRAFT_FEATURES
def test_mirror_draft_is_zero():    # blue == red picks -> every feature == 0 (lane uses 0.5 - 0.5)
def test_swap_negates():            # f(red, blue) == -f(blue, red) for every key
def test_tags_counted():            # blue has Garen (Fighter, Tank), red has Ahri -> tag_Tank_diff == 1, tag_Mage_diff == -1
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (4 passed).
- [ ] **Step 5:** Commit: `feat: draft features`

---

### Task 10: Build datasets

**Files:**
- Create: `lolwp/features/build.py`
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `Store.iter_matches`, `parse_match`, `patch_key`, `state_features`, `DraftStats`, `oof_stats`, `draft_features`, `champions.load`
- Produces CLI `python -m lolwp.features.build` writing:
  - `data/draft.parquet`: columns `match_id, patch, split ("train"/"test"), blue_win, <DRAFT_FEATURES>, blue_<role>, red_<role>` (train rows use OOF stats; test rows use full-train stats)
  - `data/state.parquet`: `match_id, patch, split, blue_win, duration_s, <STATE_FEATURES>` (only matches with a timeline and ≥ 16 frames)
  - `artifacts/draft_stats.json` (full-train stats)
  - prints row counts per split and the test patch
- Function `split_by_patch(records: list[MatchRecord]) -> tuple[list, list, str]` (train, test, test_patch)

- [ ] **Step 1: Failing tests:**

```python
def test_split_newest_patch_is_test():    # patches 15.19, 15.20, 15.21 -> test only 15.21
def test_no_test_match_in_saved_stats(tmp_path):  # build on a tiny store; DraftStats.from_json(...).match_ids ∩ test ids == ∅
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (2 passed).
- [ ] **Step 5:** After collection finishes, run `python -m lolwp.features.build`. Expected: about 18,000+ draft rows and 7,000+ state rows. If the newest patch has fewer than 1,500 draft rows, wait a few days and collect more `ids`/`details` before training.
- [ ] **Step 6:** Commit: `feat: build train/test datasets`

---

### Task 11: Explanation math

**Concept:** turning a model into "why" bars that add up exactly.

**Files:**
- Create: `lolwp/models/__init__.py`, `lolwp/models/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Produces:
  - `@dataclass Contribution(feature: str, logit: float, value_pct: float)`
  - `explain(pipeline, x: pandas.DataFrame) -> tuple[float, float, list[Contribution]]` returning `(baseline_prob, final_prob, contributions)`. `pipeline` is `make_pipeline(StandardScaler(), LogisticRegression())`. Logit contribution = `coef_i * scaled_x_i`; baseline = `sigmoid(intercept)`; sort by |logit| descending; `value_pct` = sequential probability change × 100.

- [ ] **Step 1: Failing tests** (fit a pipeline on a small synthetic dataset):

```python
def test_contributions_sum_to_final():
    base, p, cs = explain(pipe, x)
    assert base + sum(c.value_pct for c in cs) / 100 == pytest.approx(p, abs=1e-9)
def test_final_matches_predict_proba():   # p == pipe.predict_proba(x)[0, 1]
def test_sorted_by_abs_logit():
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (3 passed).
- [ ] **Step 5:** Commit: `feat: exact waterfall explanations`

---

### Task 12: Train and evaluate models

**Files:**
- Create: `lolwp/models/train.py`
- Test: `tests/test_models_sanity.py`

**Interfaces:**
- Consumes: `data/draft.parquet`, `data/state.parquet`, `DRAFT_FEATURES`, `STATE_FEATURES`
- Produces CLI `python -m lolwp.models.train` writing to `artifacts/`:
  - `draft_model.joblib`, `state_model.joblib`, `combined_model.joblib` (each a scaler + logistic regression pipeline; `max_iter=1000`)
  - `metrics.json`: for each of `baseline, draft_lr, draft_lgbm, state_lr, state_lgbm, combined_lr, combined_lgbm`: `{auc, log_loss, accuracy, brier, cv_auc_mean, cv_auc_std}` on test; plus `test_patch`, `n_train`, `n_test`, and `prev_patch_auc` for each LR (the model trained on all but the newest two patches, scored on the second-newest patch, compared with its score on the newest; this is the small drift check)
- Baseline: `sigmoid(10 * champ_wr_diff)` scored directly, no fitting.
- Combined model features: `draft_logit` + `STATE_FEATURES`. Train rows get `draft_logit` from `cross_val_predict(draft_pipeline, ..., cv=5, method="decision_function")` on the draft train rows; test rows get it from the fitted draft model.

- [ ] **Step 1: Failing tests** (skip with `pytest.mark.skipif` when `artifacts/draft_model.joblib` is missing):

```python
def test_probabilities_in_range():        # all predict_proba on test in [0, 1]
def test_draft_swap_symmetry():
    # build swapped features via draft_features(red, blue, ...) for test rows
    assert np.mean(np.abs(p + p_swapped - 1)) <= 0.05
def test_state_swap_symmetry():           # negate state features
def test_metrics_file_complete():         # all 7 model keys present with 6 metrics each
```

- [ ] **Step 2:** Run, expect SKIP (no artifacts yet). **Step 3:** Implement. **Step 4:** Run the CLI, then `pytest tests/test_models_sanity.py -v`. Expected: 4 passed.
- [ ] **Step 5:** Read `metrics.json`. Write down: draft LR vs baseline log loss, state vs combined AUC, LR vs LGBM gap. These are your headline numbers.
- [ ] **Step 6:** Commit: `feat: train and evaluate models` (artifacts included)

---

### Task 13: Statistics and insights data

**Files:**
- Create: `lolwp/models/insights.py`
- Test: `tests/test_insights.py`

**Interfaces:**
- Produces CLI `python -m lolwp.models.insights` writing `artifacts/insights.json` with keys:
  - `champions`: list of `{id, name, games, win_rate, ci_low, ci_high}` (Wilson 95% interval, champions with ≥ 30 games, train split)
  - `gold_curve`: list of `{gold_bin_center, win_rate, n}` for 1000-gold bins from −8000 to 8000 (state, all splits)
  - `objectives`: statsmodels logit `blue_win ~ gold_k + dragons_diff + towers_diff + grubs_diff + heralds_diff + kills_diff` → list of `{term, odds_ratio, ci_low, ci_high, p_value}`, plus `dragon_vs_tower: {estimate_diff, p_value}` from `t_test("dragons_diff = towers_diff")`
  - `blue_side`: `{win_rate, n, p_value}` (scipy `binomtest` vs 0.5)
  - `model_comparison`: copied from `metrics.json`
- Function `wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]`

- [ ] **Step 1: Failing tests:** `wilson(50, 100)` ≈ (0.404, 0.596) at 3 decimals; `wilson(0, 10)[0] == 0`; after running the CLI (skipif missing), every `ci_low ≤ win_rate ≤ ci_high`.
- [ ] **Step 2–4:** Run (FAIL), implement, run (pass).
- [ ] **Step 5:** Commit: `feat: statistics and insights data`

---

### Task 14: Analysis notebooks

**Concept:** this is where your thinking shows. Code without written interpretation isn't analysis.

**Files:**
- Create: `notebooks/01_eda.ipynb`, `notebooks/02_models.ipynb`, `notebooks/03_stats.ipynb`

- [ ] **Step 1:** `01_eda`: row counts per patch, blue-side win rate, win rate vs gold lead chart, win rate by first objectives, correlation heatmap of state features, top/bottom champions by win rate with intervals. After each chart, 3–5 sentences of interpretation.
- [ ] **Step 2:** `02_models`: load `metrics.json` and models; table of all 7 models; calibration plot for the three LR models; LR coefficient bar charts; the drift result; a paragraph on why LGBM did or didn't win.
- [ ] **Step 3:** `03_stats`: reproduce the objectives regression with full `summary()`, odds-ratio chart, the dragon-vs-tower test, and the "association, not causation" caveat.
- [ ] **Step 4:** Verify: Kernel → Restart & Run All on each notebook with no errors.
- [ ] **Step 5:** Commit: `docs: analysis notebooks`

---

### Task 15: API

**Files:**
- Create: `lolwp/api/__init__.py`, `lolwp/api/schemas.py`, `lolwp/api/predictor.py`, `lolwp/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: artifacts (`*_model.joblib`, `draft_stats.json`, `champions.json`, `insights.json`), `draft_features`, `explain`, `STATE_RANGES`
- Produces:
  - Schemas: `Team` (`top, jungle, mid, bot, support: int`), `PredictRequest(mode: Literal["draft","state","both"], blue: Team | None, red: Team | None, state: dict[str, int] | None)`, `ContributionOut(feature, label, value_pct)`, `PredictResponse(probability, baseline, contributions, reasons: list[str], split: {draft_pct, state_pct} | None)`
  - `class Predictor(artifacts_dir: str)` with `champions() -> list[dict]`, `predict(req: PredictRequest) -> PredictResponse`, `insights() -> dict`
  - Routes: `GET /api/champions`, `POST /api/predict`, `GET /api/insights`, `GET /api/health`
  - Validation → HTTP 422 with `{"detail": "<readable message>"}`
  - `label`: human names (e.g. `lane_mid` → "Mid lane matchup", `gold_diff` → "Gold lead at 15:00"). Reasons: top 3 contributions with templates, e.g. "Mid lane matchup favors Blue (+3.1%)".
  - Both mode uses the combined model; the draft model supplies `draft_logit`. Split per spec section 7.

- [ ] **Step 1: Failing tests** with `fastapi.testclient.TestClient` (skipif artifacts missing):

```python
def test_champions_list():                 # 200, list, each has id/name/icon_url
def test_predict_draft_ok():               # 10 valid ids -> 200, 0 <= probability <= 1, len(reasons) == 3
def test_predict_state_ok():
def test_predict_both_has_split():          # split.draft_pct + split.state_pct ≈ (probability - baseline) * 100
def test_draft_missing_team_422():
def test_both_with_partial_draft_422():     # Review Focus 4: state filled, red missing -> 422
def test_duplicate_champion_422():
def test_unknown_champion_422():
def test_state_out_of_range_422():          # Review Focus 5: gold_diff 999999 -> 422
def test_state_missing_feature_422():
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (10 passed).
- [ ] **Step 5:** Run `uvicorn lolwp.api.main:app --reload`, open `http://127.0.0.1:8000/docs`, try each endpoint by hand.
- [ ] **Step 6:** Commit: `feat: prediction API`

---

### Task 16: Frontend: Predict page

**Files:**
- Create: `web/` via Vite, `web/src/{types.ts, api.ts, App.tsx}`, `web/src/components/{ChampionPicker,DraftInput,StateInput,Footer}.tsx`, `web/src/pages/Predict.tsx`

**Interfaces:**
- Consumes: the API in Task 15 (types mirror `PredictRequest`/`PredictResponse`)
- Produces:
  - `api.ts`: `getChampions(): Promise<Champion[]>`, `predict(req: PredictRequest): Promise<PredictResponse>` (throws `Error(detail)` on 422), `getInsights(): Promise<Insights>`
  - `ChampionPicker({ value, onChange, disabledIds })`: searchable dropdown with icons; champions already picked are disabled
  - `DraftInput({ blue, red, onChange })`: two columns × 5 roles
  - `StateInput({ state, onChange })`: one slider per `STATE_FEATURES`, using the ranges from Task 1 (copied into `types.ts` as a constant)
  - `Predict` page: mode tabs (Draft / Game state / Both), Predict button disabled until required inputs are complete, shows API error text

- [ ] **Step 1:** Scaffold:

```
npm create vite@latest web -- --template react-ts
cd web && npm install && npm install react-router-dom plotly.js-dist-min react-plotly.js
npm install -D @types/react-plotly.js
```

In `web/vite.config.ts`, proxy `/api` to `http://127.0.0.1:8000` for local dev.
- [ ] **Step 2:** Implement the files above. Add routes `/` (Predict), `/insights`, `/method` in `App.tsx` (placeholder pages for the last two).
- [ ] **Step 3:** Verify by hand with the API running (`uvicorn ...`) and `npm run dev`: pick 10 champions → prediction appears; the same champion can't be picked twice; Both mode button stays disabled with a partial draft; `npm run build` succeeds with no TypeScript errors.
- [ ] **Step 4:** Commit: `feat: predict page`

---

### Task 17: Frontend: charts, Insights, Method

**Files:**
- Create: `web/src/components/{ResultPanel,WaterfallChart}.tsx`; fill `web/src/pages/{Insights,Method}.tsx`

**Interfaces:**
- Produces:
  - `WaterfallChart({ baseline, contributions })`: Plotly `type: "waterfall"`, first bar = baseline (absolute), then one relative bar per contribution, final total bar = probability
  - `ResultPanel({ result })`: large probability, waterfall, 3 reasons, and in Both mode a two-segment bar for draft vs game-state share
  - `Insights`: champion win-rate chart with error bars (searchable), gold-lead curve, odds-ratio chart with intervals, model comparison bar chart (AUC and log loss), blue-side stat
  - `Method`: data source, collection window, method choices and rejected alternatives (copy from spec section 6), limitations, link to GitHub, Riot disclaimer in the footer

- [ ] **Step 1:** Implement.
- [ ] **Step 2:** Verify by hand: bars in the waterfall add up to the shown probability; charts work on a phone-width window (Chrome devtools, 390px); no console errors.
- [ ] **Step 3:** Commit: `feat: result charts, insights and method pages`

---

### Task 18: Serve, deploy, and document

**Files:**
- Modify: `lolwp/api/main.py` (mount `web/dist` at `/` with SPA fallback to `index.html` for non-`/api` paths)
- Create: `README.md`, `render.yaml` (optional)
- Test: `tests/test_api.py` (add `test_root_serves_index` skipif `web/dist` missing)

- [ ] **Step 1:** Write the failing test: `GET /` returns 200 with `text/html`; `GET /insights` also returns the index (SPA fallback); `GET /api/health` still JSON.
- [ ] **Step 2–3:** Implement the static mount, run `npm run build` in `web/`, run `pytest -v`. Expected: all tests pass.
- [ ] **Step 4:** Push to GitHub. Confirm `.env` and `data/` are not in the repo (`git ls-files | findstr /i "env data/"` on Windows, or `grep` on Mac/Linux, shows nothing sensitive).
- [ ] **Step 5:** Render → New Web Service → connect the repo. Build command: `pip install -r requirements.txt && cd web && npm ci && npm run build`. Start command: `uvicorn lolwp.api.main:app --host 0.0.0.0 --port $PORT`. Add the Python and Node versions in environment settings if Render asks.
- [ ] **Step 6:** Verify the live URL: all three modes predict, Insights loads, Method shows the disclaimer.
- [ ] **Step 7:** Write `README.md` in this order: question, headline findings (your real numbers from `metrics.json` and `insights.json`), screenshot, live link, data collection, method and rejected alternatives, results table, drift check, limitations, how to run locally, Riot disclaimer.
- [ ] **Step 8:** Add the project to your portfolio site as the first entry.
- [ ] **Step 9:** Commit: `docs: README and deployment`

---

## Order and timing

| Day | Tasks |
|---|---|
| 1 | 1–4 (start collection overnight) |
| 2 | 5–9 while collection runs |
| 3 | Finish collection, 10–11 |
| 4 | 12–13 |
| 5 | 14 |
| 6 | 15–16 |
| 7 | 17–18 |
