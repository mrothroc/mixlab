# Grammar-Constrained Generation

Mixlab can restrict causal next-token sampling to structurally legal tokens.
Constraints run before temperature and top-k, so sampling remains stochastic
among the legal candidates. This feature applies to `-mode generate`; diffusion
generation does not have autoregressive-prefix grammar semantics.

Two grammar inputs are available:

- `-grammar-table` executes a versioned deterministic automaton over token IDs.
- `-grammar` or `-grammar-string` executes a GBNF grammar over decoded token
  bytes.

Only one grammar input may be supplied. Constrained generation requires an EOS
token through `-eos-token-id` or `-sequence-vocab`.

## Token-DFA Tables

Token-DFA tables are tokenizer-specific and do not require Mixlab to decode
token text. Missing transitions are forbidden. Transitions are the only source
of legal-token information, avoiding separate allowed-token lists that can
disagree with state transitions.

```json
{
  "format": "mixlab.token_dfa",
  "version": 1,
  "vocab_size": 5,
  "start_state": "start",
  "eos_token_ids": [4],
  "states": [
    {"name": "start", "transitions": {"0": "body"}},
    {"name": "body", "transitions": {"1": "open", "3": "body", "4": "done"}},
    {"name": "open", "transitions": {"2": "body", "3": "open"}},
    {"name": "done", "accept": true}
  ]
}
```

This example treats token `0` as BOS, `1`/`2` as parentheses, `3` as content,
and `4` as EOS. Every EOS transition must target an accepting state. The CLI
EOS ID must be listed in `eos_token_ids`.

Tables are deliberately finite-state and domain-neutral. A producer can compile
bounded nesting depths, quote modes, or finite label/register combinations into
states. Mixlab does not interpret domain-specific stack or counter actions.
Version 1 limits tables to 100,000 states and 1,000,000 transitions.

```bash
./mixlab -mode generate \
  -config model.json \
  -safetensors-load weights.safetensors \
  -prompt token_ids:0 \
  -eos-token-id 4 \
  -grammar-table examples/grammars/balanced_parentheses.token_dfa.json
```

## GBNF

GBNF constraints operate on bytes. For normal text models, pass a Hugging Face
ByteLevel BPE `tokenizer.json`; Mixlab also supports its nucleotide vocabulary
artifact. BPE tokens are checked as complete byte sequences, including tokens
that represent only a partial grammar construct.

```gbnf
root ::= object
object ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= "\"value\"" ws ":" ws number
number ::= "-"? [0-9]+
ws ::= [ \t\n\r]*
```

Supported syntax includes named rules, `|`, grouping, string literals, `.`,
ASCII byte classes and ranges, `?`, `*`, `+`, and `{m}`, `{m,n}`, `{m,}`.
String literals support UTF-8 and common escapes. Character classes are
byte-oriented and therefore limited to ASCII or `\xNN` byte values. Grammars
whose expansion is left-recursive or exceeds the documented runtime limits are
rejected instead of consuming unbounded memory.

```bash
./mixlab -mode generate \
  -config model.json \
  -safetensors-load weights.safetensors \
  -tokenizer-path data/tokenizer.json \
  -prompt token_ids:1 \
  -eos-token-id 2 \
  -grammar examples/grammars/json_object.gbnf
```

## Prompt Semantics

`-grammar-prompt-mode consume` is the default. Every non-special prompt token
must advance the grammar, while tokenizer special tokens such as BOS are
ignored by GBNF and represented explicitly in token-DFA tables. This mode is
appropriate when the prompt is a prefix of the constrained value.

Use `-grammar-prompt-mode ignore` when the prompt is model context and the
grammar applies only to the generated continuation. Output still contains the
prompt, so structural-validity claims apply to the continuation in this mode.

An empty Mixlab prompt seeds causal generation with a random token. Use an
explicit BOS/prefix whenever the complete emitted token record must satisfy the
grammar.

## Completion And Errors

EOS is permitted only when the grammar accepts. A token-DFA must encode EOS as
an explicit transition to an accepting state; GBNF exposes the configured EOS
token when its root rule accepts. Mixlab returns an error when:

- no legal finite logit remains;
- a prompt does not match in `consume` mode;
- model logits contain `NaN` or positive infinity;
- the grammar exceeds state/stack limits; or
- generation reaches a configured length limit before the grammar accepts.

The default `-grammar-on-incomplete=error` preserves this fail-fast behavior.
For bulk sampling, use:

```bash
./mixlab -mode generate \
  -config model.json \
  -safetensors-load weights.safetensors \
  -prompt token_ids:0 \
  -eos-token-id 4 \
  -grammar-table grammar.json \
  -grammar-on-incomplete skip \
  -num-samples 10000 \
  -gen-batch 64 \
  -generate-out samples.txt
```

In `skip` mode, `-num-samples` counts accepted outputs, not attempts. Mixlab
discards a sample only when its processor reports that the grammar is still
non-accepting after `max-tokens` or `seq_len` exhaustion. Prompt errors,
invalid logits, grammar resource-limit failures, and states with no legal
continuation remain fatal because they may indicate a bad grammar or runtime
failure.

`-grammar-max-attempts` bounds the total work. Its default `0` resolves to
`4 * num-samples`; an explicit value must be at least `num-samples`. If the cap
is exhausted, Mixlab keeps already-written accepted records but exits nonzero
with a shortfall error. Attempt indices, rather than accepted-output indices,
own RNG streams. The accepted output is therefore deterministic for a fixed
seed and independent of `gen-batch` on the same backend and hardware.

At completion, Mixlab writes accepted records only to `generate-out` and a
summary to stderr. There is intentionally no `truncate` policy: emitting a
non-accepting prefix through the normal sample stream would weaken the
structural-validity contract and require a separately marked output schema.
