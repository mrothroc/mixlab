# data/ — shard loaders, binary formats, dataset manifest

Pure Go (no cgo). Turns on-disk shards into fixed-shape training `Batch`es and
validates the dataset↔model contract. Grew into a core subsystem with the
sequence-modality work (nucleotide, record framing, classification).

User-facing guide: [`../docs/data.md`](../docs/data.md) (formats, prepare flags,
manifest fields). Prep scripts that *write* these formats live in `../scripts/`.

## Key files
- `loader.go` — `Loader`/`NextBatchDetailed`; picks the stream by manifest. `Batch` carries optional `LossMask`, `SegmentIDs`, `MaskEligible`, `Labels`, `ValidMask` (all nil for the legacy flat path, preserving it exactly).
- `manifest.go` — `mixlab.dataset.json` (`DatasetManifest`): representation, shard format, sequence layout, `special_token_ids`, `Task.num_labels`. Strict load (unknown-field + trailing-JSON rejection); cross-checks model vocab. `EffectiveSequenceLayout()` keeps pre-layout manifests backward-compatible.
- `sequence_shard.go` — record-oriented readers with exact-size + terminal-offset validation (malformed shard → error, never OOB/panic).
- `nucleotide.go` — `NucleotideVocabulary`: DNA/RNA, IUPAC ambiguity, complement tables (used by reverse-complement augmentation).

## Shard formats (magic / version)
| Format string | magic | Reader | Payload |
|---|---|---|---|
| `mixlab_token_shard_v1` | 20240520 | `LoadDataShard` | flat uint16 token stream |
| `mixlab_sequence_shard_v1` | 20260718 | `LoadSequenceShard` | offsets table + uint16 records |
| `mixlab_labeled_sequence_v1` | 20260724 | `LoadLabeledSequenceShard` | offsets + int32 labels + uint16 records |

All little-endian with a 256-int32 header. Sequence layouts: `packed_segments`
(pack many records per row with BOS/EOS) and `one_record_per_row` (one framed
record per row, used by record framing + classification).

## Conventions
- The Python writer (`scripts/prepare*.py`) and the Go reader are a **byte-exact contract** — change both together and keep the magic/version, header size, field order, and endianness in lockstep.
- A new shard format bumps a new magic constant; never silently repurpose an existing one.
- Label/token range checks belong on an authoritative path that runs **before** any indexing (prepare, manifest, and a runtime guard) — see GitHub issue #2 for the resume-hash follow-up.
