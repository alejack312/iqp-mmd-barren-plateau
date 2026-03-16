# Upstream consistency audit: local clone vs XanaduAI/scaling-gqml

Performed 2026-03-16. Compared the local clone at `/tmp/scaling-gqml-src` against commit [`b907bb1`](https://github.com/XanaduAI/scaling-gqml/tree/b907bb119c45ee85c87f1eb91867f4a7d281f5be) on GitHub.

## The short version

The local clone is the upstream repo at its current HEAD (`8e5620c`). No local
modifications, no fork — it's a straight clone of `XanaduAI/scaling-gqml`. The
target commit `b907bb1` is simply an older point in that same history, 7 commits
back. Seven files differ because the repo kept evolving after `b907bb1` was
created.

| | Local | Target |
|---|---|---|
| Commit | `8e5620c` ("repickle params") | `b907bb1` ("Update README.md") |
| Working tree clean? | Yes | n/a |
| Relationship | `b907bb1` is an ancestor of local HEAD | — |

## Commit history

After fetching the full history, the ancestry is linear (with one merge):

```
8e5620c  ← local HEAD, current remote HEAD
657dd90  Update README.md
6ccb4a0  Update README.md
6c259e9  Update README.md
6dfc0df  Update README.md
c571d84  code for barren plateaus
715ed6e  Merge branch 'main' of https://github.com/XanaduAI/scaling-gqml
├── b907bb1  ← target commit (this audit's reference point)
└── 160cf91  with spin sym
e577848  add code
...
```

`b907bb1` was on a branch that got merged into `main` at `715ed6e`. Everything
from `715ed6e` onward is the 7 commits that explain the file differences below.
The initial audit incorrectly described this as a "divergence" — the local is
not a fork, it's just newer.

## What differs (7 files)

### README.md

Local has the published version with the real arxiv reference (`2503.02934`), 
a contact email, and clean dependency links. Target still has `XXXX:XXXX` as 
the arxiv placeholder and points dependencies at specific branch paths instead 
of the repo root.

Direction of drift: local is ahead.

### paper/README.md

The imports are organized differently. Target groups all imports 
(`sys.path.append`, `from iqpopt import *`, model imports) into a single block 
at the top of the code example. Local distributes them into each model's section.

There's also an API change: target calls `model.sample(params_iqp, 100)` 
(positional), local calls `model.sample(params_iqp, shots=100)` (keyword). This 
likely reflects a change in `iqpopt`'s API between versions.

### paper/plots/calc_grads.py (130 lines) and paper/plots/plot_grads.py (85 lines)

These two files exist locally but are absent from the target. They calculate and 
plot gradient norms. Either they were added after `b907bb1` was tagged, or they 
were removed from that branch and kept in the local fork. Given the commit 
messages, the local repo added them later.

### paper/plots/samples/samples-IqpSimulatorBitflip_2D_ising.csv

About 20 rows of sample data differ. The values are binary spin configurations, 
so this is almost certainly a re-sampling with a different seed or a retrained 
model. Not a formatting issue; genuinely different data.

### paper/plots/tables/2D_ising-mmd_loss.csv

This one matters. The numerical results are completely different, not just rounding:

- Target has positive IqpSimulator loss values (e.g. `7.93e-05` at sigma=0.6)
- Local has negative values (e.g. `-1.005e-04` at sigma=0.6)
- Same pattern for IqpSimulatorBitflip: target shows `0.0161` where local shows `-3.14e-06`

The row ordering also changed. Target inserts the IqpSimulator and 
IqpSimulatorBitflip blocks between the existing models; local appends them at 
the end with different values.

This is the most significant drift. If the local values fed into any downstream 
analysis or plots in our project, the results won't match the published repo.

### paper/training/trained_parameters/params_IqpSimulator_genomic-805.pkl

Binary pickle file, 16 bytes different. This is the "repickle params" commit -- 
the parameters were re-serialized (possibly to fix a Python version or pickle 
protocol mismatch). The actual parameter values may or may not have changed; 
without loading both pickles and comparing arrays we can't tell from the diff 
alone.

## What matches

Everything else. All 150+ other files across datasets, hyperparameter search 
results, other training scripts, benchmark data, loss plots, and other trained 
parameter pickles are byte-identical between local and target.

## What to do about it

Depends on which version we want as our reference:

1. **Check out the target commit.** Run `git checkout b907bb1` in the local
clone to pin to that exact state. This drops the gradient scripts, reverts the
READMEs to pre-publication, and swaps in the older MMD loss values. Useful if
you need to reproduce results from that specific snapshot.

2. **Keep local as-is.** The local HEAD has the published paper's README, the
extra gradient scripts, and the latest parameter files. If our project references
the arxiv paper, this is the better source of truth — it's what the authors
ship today.

3. **Cherry-pick specific files.** If we only care about the training scripts 
and parameters (the `paper/training/` subtree), the only drift there is the 
re-pickled genomic params file. Everything else in `paper/training/` is identical.
