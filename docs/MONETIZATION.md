# Launch kit

Copy for getting ML Workbench in front of people who have the problem. Everything here is
written to be posted as-is; edit the numbers only if you change the pricing.

The rule that matters: **post where the pain is, not where the programmers are.** A generic
"check out my ML tool" post on r/MachineLearning dies in minutes. "Here is how I stopped
trusting R² scores from point-and-click tools" does not.

---

## 1. Show HN

**Title** (HN titles are the whole game — no adjectives, no product name first):

> Show HN: A desktop ML tool that shows you the baseline and keeps an honest holdout

**Body:**

```
I built this after watching people use no-code ML GUIs and come away with a number they
couldn't defend. The failure modes were always the same, so each one is addressed explicitly
rather than left to the user:

- Every run trains a baseline (DummyRegressor / majority class) and pins it in the results
  table, so R2 = 0.72 has something to mean.
- Every model is a full sklearn Pipeline — impute -> one-hot -> scale -> estimator — pickled
  as one object. Train/predict can't drift apart because preprocessing travels with the model.
- An optional holdout is never used for tuning and is reported in its own columns, so you can
  see how optimistic the CV number was.
- Model selection uses the 1-SE rule: it recommends the simplest model within one standard
  error of the best, and says why in a sentence.
- Every run writes run.json (settings, seed, library versions, sha256 of the input data),
  report.md, metrics.csv, plots/ and the pickled pipeline.

Also: TimeSeriesSplit + group-aware CV, a lag/rolling feature builder, permutation importance
that works with named columns, applicability-domain checks at prediction time (out-of-range
rows flagged, unseen categories counted), SMILES -> RDKit descriptors/fingerprints with salt
stripping, and a range-based optimiser that warns you when the optimum is on a boundary or
outside the training range.

Single 113 MB Windows exe, tkinter GUI, no network code at all — it runs on an air-gapped
machine, which is the actual constraint in a lot of labs. Same engine on the CLI
(mlw train / predict / optimise / audit / smiles). MIT.

There's an 82-page guide written for people with no ML background, because the audience I
care about is a chemist or a process engineer with a CSV, not a data scientist.

Happy to answer anything about the design — particularly the selection rule, which is the
part I have the strongest opinions about.
```

**First comment to post immediately after submitting** (HN readers judge on responsiveness):

```
Author here. Two deliberate limitations worth naming before anyone finds them:

1. Hyperparameter search and scoring share folds, so CV scores are mildly optimistic. The
   holdout column is the unbiased number — opt-in, but strongly suggested.
2. Trees, fingerprints and descriptor models cannot extrapolate. The app warns you at
   prediction time rather than pretending otherwise; it can't fix it.

If you want custom feature engineering, text/image models, survival analysis or Bayesian
inference, you want a notebook — this covers the screening/comparison/prediction 80%.
```

---

## 2. r/MachineLearning

Flair: `[P]` (project). Self-promo rules are strict, so lead with the design argument.

**Title:**

> [P] ML Workbench — a desktop tool where every run has a baseline, an untouched holdout, and a data hash

**Body:** use the HN body above, then add:

```
Repo: <link>   Website: <link>   Windows exe in Releases (MIT)

What I'd really like feedback on: the 1-SE selection rule as a default. It regularly picks
RandomForest over GradientBoosting when the difference is inside the noise, which makes some
people uncomfortable because the leaderboard no longer has an obvious winner. I think that's
correct behaviour for a screening tool. Curious whether people who teach ML agree.
```

---

## 3. r/chemistry and r/cheminformatics

Different angle: the molecule workflow, not the ML theory.

**Title:**

> A free offline tool that turns a SMILES column into RDKit descriptors/fingerprints and then models it, with applicability-domain checks

**Body:**

```
I kept seeing the same two problems in QSAR-ish work done by non-programmers:

1. The descriptor step and the model step live in different tools, so nobody can reproduce
   the pair later.
2. Predictions get made on molecules far outside the training set, and nothing says so.

ML Workbench does both steps in one place. SMILES in -> salts/mixtures reduced to the largest
fragment, canonical SMILES, 15 RDKit descriptors or Morgan/RDKit fingerprints -> train with a
baseline and an honest holdout -> predict on new molecules with a Tanimoto similarity check
against the training set, so you get a warning instead of a confident number about a scaffold
nothing like your training data.

Unparsable SMILES are reported, not silently blanked. Every run writes run.json including the
sha256 of the input, so the descriptors+model pair is reproducible.

Free, MIT, Windows exe in Releases, completely offline (no network code). 82-page guide
included. Standardisation is deliberately simple — largest fragment, canonical SMILES, no
tautomer or charge normalisation — so if that matters for your chemistry, use a proper
cheminformatics pipeline for that step.
```

---

## 4. LinkedIn / X thread

Short, first-person, no hashtag soup. The hook is the *problem*, not the app.

```
A no-code ML tool gave a colleague R² = 0.72.

I asked what the baseline was. Blank look. We retrained with a DummyRegressor in the line-up:
R² = -0.14. The "model" was better than guessing, but nowhere near as good as it looked.

That exchange is why I built ML Workbench. Every run now:

- trains a baseline and pins it in the results table
- keeps an untouched holdout for the unbiased number
- recommends the simplest model within one standard error of the best
- writes run.json with the seed, library versions and sha256 of the data

Free, MIT, offline, Windows. 113 MB, nothing to install.
<links>
```

Then one follow-up post per feature with a single screenshot. Screenshots travel; prose doesn't.

---

## 5. Direct outreach (this converts better than any post)

Find people publicly asking the question this solves and answer it *for free*, with the tool
as an aside. Search terms that surface real demand:

- `"one hot encoder" pickled model prediction wrong site:stackoverflow.com`
- `"how do I know if my model is good" csv excel`
- `"random forest" "excel" lab yield prediction` on Reddit
- GitHub issues on `EZmachinelearning`-style repos complaining about missing values or no baseline

Reply with the actual fix first. Mention the tool only if it genuinely applies, once. Do not
paste the same reply in ten threads — that's how accounts get banned and how the project gets
a reputation for spam.

**Where the paying customers actually are:** process engineers, analytical chemists, QC labs,
and small manufacturers with an Excel file and a yield problem. They are on LinkedIn and in
vendor forums, not on Hacker News. One post in a coatings or fermentation LinkedIn group is
worth more than a front page on HN for revenue — HN is worth more for contributors and
credibility.

---

## 6. Getting to the first dollar

Ordered by effort-to-likelihood, based on how these things actually convert:

| Step | Effort | Why it works |
|---|---|---|
| Publish the site, pin the repo, screenshot in the README | 1 hour | Without a landing page nothing downstream happens |
| Answer 5 real questions where the tool applies | 2 hours | Traffic that arrives with a problem converts; traffic from a launch post mostly doesn't |
| Add "paid help" links + a pricing section | done | Removes the "how would I even pay this person" friction |
| One LinkedIn post aimed at a specific industry | 30 min | The first genuine enquiries usually come from here, not HN |
| Apply for GitHub Sponsors | 10 min + review | Low value early, compounding later; Egypt is a supported region |
| Write one case study with real numbers | 3 hours | The single most persuasive asset for the $250 tier |

---

## 7. What not to do

- **Don't put the exe behind a paywall.** It's MIT and already published; reversing that
  costs the credibility that makes the paid tiers viable.
- **Don't add telemetry to "measure engagement."** The no-network property is a genuine
  selling point for lab and clinical users. It's worth more than the analytics.
- **Don't launch on five platforms in one day.** You can only answer comments on one. HN
  rewards the first 90 minutes; a stale submission with unanswered questions reads as dead.
- **Don't lead with "AI".** The audience for this is buying defensible numbers, not novelty.
