# These four files are hand-authored. No model produced them.

Every line in this directory was typed by hand to exercise a scorer. Nothing
here is captured output: no provider was called, no telemetry was read, no
classifier was run. The word "recording" names the shape of the file, which is
what `RecordedClient` and `RecordedClassifier` replay, not where it came from.

| File | Replayed by | What it is |
|---|---|---|
| `dev.jsonl` | `RecordedClient` | 21 `{prompt, system, response}` rows, one per dev case |
| `holdout.jsonl` | `RecordedClient` | 17 rows, one per held-out case |
| `dev-toxicity.jsonl` | `RecordedClassifier` | 4 `{text, labels}` rows, the label scores a head would return |
| `holdout-toxicity.jsonl` | `RecordedClassifier` | 3 rows, same shape |

The toxicity scores are the ones a case needs in order to pass or fail on the
side it is written to test. They are not an estimate of how a real head would
score that sentence, and they should not be read as one.

Two consequences worth knowing before you trust a number out of this suite:

- **A response here is the response the case was written around.** Exactly four
  of the 38 score 0.0: `g-05` in dev, and `hf-01`, `hg-04` and `hr-01` in the
  held-out split. Two of those four trip a threshold and are the held-out
  failures the top-level README quotes; the other two are absorbed by a
  groundedness mean, which is the per-case gap recorded under "Not done". The
  suite measures the scorers against known answers. It does not measure a model.
- **A key is the exact prompt text.** `RecordedClient` keys on the first 16
  characters of `sha256(prompt, system)`, with no normalisation, so editing a
  prompt in `suites/dev/` invalidates its row here and the run stops with the
  missing key rather than scoring something stale.

To put real completions behind the same interface, capture them to a new file
and leave these alone:

```
pip install -e ".[anthropic]"
llm-evals record --split dev --client anthropic --out suites/recordings/local-dev.jsonl
```

`local-*.jsonl` is gitignored, so a capture never silently replaces a fixture
this README describes as hand-written.
