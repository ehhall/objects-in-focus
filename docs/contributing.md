# Contributing

Bug reports, corrections to the data, and new tools built on top are all
welcome.

## Setting up

```bash
git clone https://github.com/ehhall/objects-in-focus.git
cd objects-in-focus
pip install -e ".[dev]"
pytest
```

The test suite runs in a few seconds. Most tests build a miniature synthetic
dataset in a temp folder, so they pass without the large data files; the ones
marked `needs_real_data` skip automatically when the published data is not
present, and run when it is.

```bash
pytest -q                 # everything available
ruff check src tests      # lint
```

## What is useful

**Data corrections.** If a polygon is wrong, a label is misspelled, or an
object was missed, open an issue with the scene name and what you saw. The
segmentations are hand-made and 2,870 polygons is a lot of hand.

**A new fixation format.** If your eye-tracker's export is not recognised, add
it to `SCHEMAS` in `oif/fixations.py` — it is a dict of canonical name to
source column — and a test with a few rows of that layout.

**A better assignment rule.** `point`, `disc` and `nearest` are three
reasonable answers to a question with more than three. Anything principled
that takes gaze uncertainty seriously would be a real addition.

**Salience models.** The package deliberately does not bundle one. A thin
adapter that computes maps from a current model on demand would save everyone
a step.

## House style

- Public functions get a docstring that says what the thing is for, not just
  what its arguments are.
- Comments explain *why*, when why is not obvious. No comment that restates
  the line below it.
- New behaviour comes with a test. New data behaviour comes with a test that
  would fail on the old behaviour.
- Errors should say what to do next. `"no annotations/ file for scene 'x'"`
  beats a bare `KeyError`.
- No breaking changes to the canonical fixation schema or the array
  conventions without a note in `CHANGELOG.md`.

## Pull requests

Small and focused travels faster than large and comprehensive. Say what
problem the change solves; if it changes numbers anyone might have published,
say so explicitly in the description.

## Reporting a data problem

Please include:

- the scene name
- what you ran
- what you expected and what you got
- `oif check` output, if it is about a missing or damaged file
