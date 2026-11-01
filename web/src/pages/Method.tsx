// Set this to the public repository URL once the project is pushed.
const REPO_URL = 'https://github.com/Joooban/Draftcast'

export default function Method() {
  return (
    <div className="page prose">
      <h1>Method</h1>
      <p className="lead">
        How much does the draft decide a League of Legends game, compared with the first 15 minutes? Published League
        prediction work is several years old, and the early game has changed since (voidgrubs, Atakhan, objective and
        tower reward changes). This project asks the question again on current games.
      </p>

      <h2>Data</h2>
      <ul>
        <li>
          Source: the official Riot Games API with the author's developer key. Only public ranked match data is used;
          no one's account or login.
        </li>
        <li>
          Population: ranked solo/duo (queue 420) on the SG2 server, Emerald and above. Players are seeded from the
          Emerald, Diamond, Master, Grandmaster and Challenger ladders.
        </li>
        <li>Window: matches from the last 42 days at collection time, about three patches.</li>
        <li>
          Cleaning: remakes and early surrenders are dropped, as are matches where any player lacks a clear role.
          Game-state rows also need the game to last at least 16 minutes.
        </li>
        <li>
          Split by time: the newest patch is the test set; all older patches are training data. Tuning uses 5-fold
          cross-validation inside the training set only.
        </li>
      </ul>

      <h2>Features</h2>
      <ul>
        <li>
          <strong>Draft:</strong> difference in average champion win rate, the win rate of each lane matchup, bot duo
          synergy, and team composition from Data Dragon (class tags, physical vs magic damage). Win rates are shrunk
          toward 50% with a prior of 20 games, so rare champions and matchups don't get extreme values; unseen ones
          get exactly 50%.
        </li>
        <li>
          <strong>No leakage:</strong> a training match's own result never feeds the win rates that describe it. Draft
          features for training rows use 5-fold out-of-fold statistics. Test rows and this app use statistics from
          the full training set.
        </li>
        <li>
          <strong>Game state at 15:00:</strong> gold, XP, CS, level, kills, dragons, voidgrubs, towers,
          plates (inner turrets have plates too), wards placed and cleared, and who got first blood, dragon and tower. All are Blue minus Red.
        </li>
        <li>
          <strong>Combined:</strong> the draft model's log-odds output (out-of-fold for training rows) plus the
          game-state features.
        </li>
        <li>Never used: game length and any end-of-game statistic.</li>
      </ul>

      <h2>Models</h2>
      <p>
        Each mode uses a logistic regression on standardized features. In log-odds every feature adds exactly{' '}
        <em>coefficient × value</em>, so the “why” chart is exact: start from the average game, add features in order
        of size, convert to probability after each step, and the bars sum to the prediction. The baseline for
        comparison maps the champion win-rate difference straight to a probability with no fitting.
      </p>

      <h2>Alternatives considered and rejected</h2>
      <ul>
        <li>
          <strong>LightGBM as the main model:</strong> explanations would be approximate (SHAP) and the expected
          accuracy gain is small. It is trained as a benchmark; if it beats logistic regression by more than 0.01 AUC
          that is reported, but the explainable model still runs the app.
        </li>
        <li>
          <strong>Neural networks</strong> (champion embeddings, transformer draft models): need far more data and are
          hard to explain. Future work.
        </li>
        <li>
          <strong>Random forest:</strong> usually worse than LightGBM and no easier to explain.
        </li>
        <li>
          <strong>Player-history features:</strong> would need ten Riot IDs per prediction and many more API calls.
          Out of scope.
        </li>
      </ul>

      <h2>Limitations</h2>
      <ul>
        <li>One server and rank band. Results may not carry over to other regions or lower ranks.</li>
        <li>About three patches of data; champion and matchup estimates for rare picks lean heavily on the 50% prior.</li>
        <li>Balance changes between patches shift win rates; the time-based split measures how much that hurts.</li>
        <li>
          The objective statistics show association, not causation: teams that take dragons may differ in ways the
          model doesn't see.
        </li>
        <li>No player skill, champion mastery, or live data is used.</li>
      </ul>

      {REPO_URL && (
        <p>
          Code, notebooks and tests: <a href={REPO_URL}>{REPO_URL}</a>
        </p>
      )}
    </div>
  )
}
