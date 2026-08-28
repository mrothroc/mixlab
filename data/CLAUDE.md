# data/ — shard loaders, binary formats, dataset manifest

Pure Go (no cgo). Turns on-disk shards into fixed-shape training `Batch`es and
validates the dataset↔model contract. Grew into a core subsystem with the
sequence-modality work (nucleotide, record framing, classification).

User-facing guide: [`../docs/data.md`](../docs/data.md) (formats, prepare flags,
manifest fields). Prep scripts that *write* these formats live in `../scripts/`.

## Key files
- `loader.go` — `Loader`/`NextBatchDetailed`; picks the stream by manifest. `Batch` carries optional `LossMask`, `SegmentIDs`, `MaskEligible`, `Labels`, `ValidMask`, `Frames`, `Codebooks` (all nil for the legacy flat path, preserving it exactly), plus `ExampleMask`/`SeqLen`/`BatchSize`/`ExampleCount` for length bucketing. Zero `SeqLen`/`BatchSize` means "use the caller's default shape".
- `length_buckets.go` — `LoaderOptions.LengthBuckets` groups records by valid length. With `LengthBucketBatchSize` set, remainders pack across buckets *and shards*, so one batch can draw rows from several shards; rows stay grouped by source so the streams' single-slot shard cache holds one shard at a time.
- `manifest.go` — `mixlab.dataset.json` (`DatasetManifest`): representation, shard format, sequence layout, `special_token_ids`, `Task.num_labels`. Strict load (unknown-field + trailing-JSON rejection); cross-checks model vocab. `EffectiveSequenceLayout()` keeps pre-layout manifests backward-compatible.
- `sequence_shard.go` — record-oriented readers with exact-size + terminal-offset validation (malformed shard → error, never OOB/panic).
- `nucleotide.go` — `NucleotideVocabulary`: DNA/RNA, IUPAC ambiguity, complement tables (used by reverse-complement augmentation).

## Shard formats (magic / version)
| Format string | magic | Reader | Payload |
|---|---|---|---|
| `mixlab_token_shard_v1` | 20240520 | `LoadDataShard` | flat uint16 token stream |
| `mixlab_sequence_shard_v1` | 20260718 | `LoadSequenceShard` | offsets table + uint16 records |
| `mixlab_labeled_sequence_shard_v1` | 20260724 | `LoadLabeledSequenceShard` | offsets + int32 labels + uint16 records |
| `mixlab_continuous_sequence_shard_v1` | 20260726 | `LoadContinuousSequenceShard` | int32 labels + `[N,T,F]` float32 frames |
| `mixlab_continuous_sequence_shard_v2` | 20260726 | `LoadContinuousSequenceShard` | adds an int32 valid length per record, between labels and frames |
| `mixlab_codebook_sequence_shard_v1` | 20260819 | `LoadCodebookSequenceShard` | int32 labels + valid lengths + `[N,T,Q]` int32 codes |

Continuous v1 and v2 share a magic and are told apart by the header version
field, so the reader accepts both and treats v1 records as fully valid. Every
other format gets its own magic.

All little-endian with a 256-int32 header. Sequence layouts: `packed_segments`
(pack many records per row with BOS/EOS) and `one_record_per_row` (one framed
record per row, used by record framing + classification).

## Record order is a correctness concern

Preparation picks train/validation members per label but preserves source order
within each split — it does not shuffle. Class-ordered input therefore yields
class-ordered shards, and when a shard holds one class neither shard-order nor
within-shard shuffling can make that shard diverse. Most batches remain
class-pure, while batches that straddle shard boundaries can contain a few
labels. Global `class_counts` stay perfectly balanced either way, so the
manifest cannot reveal it. `train/classification_data_diagnostics.go` samples
the batches the trainer actually consumes and warns on low mean diversity;
boundary batches do not suppress the warning. Reproducing this needs a fixture
spanning **more than one shard**; the same records in a single shard train with
mixed batches.

## Conventions
- The Python writer (`scripts/prepare*.py`) and the Go reader are a **byte-exact contract** — change both together and keep the magic/version, header size, field order, and endianness in lockstep.
- A new shard format bumps a new magic constant; never silently repurpose an existing one.
- Label/token range checks belong on an authoritative path that runs **before** any indexing (prepare, manifest, and a runtime guard) — see GitHub issue #2 for the resume-hash follow-up.
