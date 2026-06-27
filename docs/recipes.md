# Recipes

A registry of real, reproducible mixlab training runs that produced a published
model. Each recipe lives in its own repository — pinned to the mixlab version it
was validated against — so it stays reproducible even as the engine evolves.
This page is the index; the recipe repos hold the configs, data-prep and eval
scripts, results, and Hugging Face links.

> Looking for small, runnable feature demos instead? See [`examples/`](../examples/README.md)
> (tiny configs that exercise individual features) and [`experiments/`](../experiments)
> (committed test configs). Those are engine-internal; recipes below are full,
> published training runs.

## Registry

| Recipe | Task | Architecture highlights | mixlab | Hardware / time | Links |
|--------|------|-------------------------|--------|-----------------|-------|
| BabyLM 2025 GPT-BERT replica | 10M-word pretraining, ~33M params | DeBERTa-style disentangled relative attention, GeGLU, dense layer aggregation, masked + causal hybrid training | `v0.33.3` | M1 Max, ~5.5h | [recipe](https://github.com/mrothroc/mixlab-babylm-gptbert) · [HF model](https://huggingface.co/mrothroc/mixlab-gptbert-masked-focus-replica) |

Selected scores for the BabyLM replica: BLiMP 70.64, BLiMP-supplement 61.79,
EWoK 50.88, entity tracking 40.33, COMPS 52.85, reading (eye+SPR) 7.30,
GLUE 64.18. See the recipe repo for the full per-component table and the
reference comparison.

## What a recipe is

A recipe is a self-contained kit for reproducing one published model. The
canonical shape (see the BabyLM repo above as a template) is:

- **Config** — the exact `ArchConfig` JSON used to train the model.
- **Data prep** — the command(s) to fetch and tokenize/pack the corpus
  (mixlab's `prepare` mode and/or a small script), naming the data source.
- **Train command** — the full `mixlab -mode arch …` invocation, including
  hardware and wall-clock time.
- **Eval** — the scripts and reported metrics, ideally against a reference.
- **README** — ties it together and **pins the mixlab version** it was validated
  against (release tag or commit).
- **Links** — the resulting Hugging Face model, and back to this engine.

Recipes deliberately **do not** vendor data shards or model weights — link out to
the corpus source and the Hugging Face model instead, so the repo stays small.

The mixlab version pin matters: the engine moves (config schema, CLI flags, and
defaults change between releases), so a recipe is reproducible against the
version it names, not necessarily the latest `main`. The `mixlab` column above
records that version for each entry.

## Contribute a recipe

1. Build your recipe as a standalone repository following the shape above. The
   [BabyLM GPT-BERT recipe](https://github.com/mrothroc/mixlab-babylm-gptbert)
   is a good template.
2. Pin the mixlab release tag (or commit) you validated against in its README.
3. Open a pull request adding one row to the Registry table here, with links to
   your recipe repo and the published Hugging Face model.
