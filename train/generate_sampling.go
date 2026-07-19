package train

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"

	"github.com/mrothroc/mixlab/data"
)

const generationSeedMix uint64 = 0x9e3779b97f4a7c15

type generationWriter interface {
	io.Writer
}

type causalGenerationEvaluator interface {
	EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error)
	ReadOutput(name string, shape []int) ([]float32, error)
}

type generationPlan struct {
	numSamples   int
	baseSeed     int64
	eosTokenID   int
	legacyOutput bool
}

type generationSample struct {
	tokens []int
	notice string
}

type generationOutput struct {
	writer *bufio.Writer
	file   *os.File
}

func openGenerationOutput(path string) (*generationOutput, error) {
	if path == "" {
		return &generationOutput{writer: bufio.NewWriter(os.Stdout)}, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create generation output %q: %w", path, err)
	}
	return &generationOutput{writer: bufio.NewWriter(file), file: file}, nil
}

func (o *generationOutput) Close() error {
	if o == nil {
		return nil
	}
	flushErr := o.writer.Flush()
	var closeErr error
	if o.file != nil {
		closeErr = o.file.Close()
	}
	return errors.Join(flushErr, closeErr)
}

func buildGenerationPlan(opts GenerateOptions, cfg *ArchConfig, vocab *data.NucleotideVocabulary) (generationPlan, error) {
	if cfg == nil {
		return generationPlan{}, fmt.Errorf("nil generation config")
	}
	if opts.MaxTokens < 0 {
		return generationPlan{}, fmt.Errorf("-max-tokens must be >= 0")
	}
	if opts.Temperature <= 0 {
		return generationPlan{}, fmt.Errorf("-temperature must be > 0")
	}
	if opts.TopK < 0 {
		return generationPlan{}, fmt.Errorf("-top-k must be >= 0")
	}
	numSamples := opts.NumSamples
	if numSamples == 0 {
		numSamples = 1
	}
	if numSamples < 1 {
		return generationPlan{}, fmt.Errorf("-num-samples must be >= 1")
	}
	eosTokenID := -1
	if vocab != nil {
		var ok bool
		eosTokenID, ok = vocab.SpecialTokenID("eos")
		if !ok {
			return generationPlan{}, fmt.Errorf("-sequence-vocab has no EOS token")
		}
	}
	if opts.EOSTokenID != nil {
		requested := *opts.EOSTokenID
		if requested < -1 || requested >= cfg.VocabSize {
			return generationPlan{}, fmt.Errorf("-eos-token-id=%d must be -1 or in [0,%d)", requested, cfg.VocabSize)
		}
		if requested >= 0 {
			if vocab != nil && requested != eosTokenID {
				return generationPlan{}, fmt.Errorf("-eos-token-id=%d does not match -sequence-vocab EOS id %d", requested, eosTokenID)
			}
			eosTokenID = requested
		}
	}
	baseSeed := opts.GenerationSeed
	if baseSeed == 0 {
		baseSeed = cfg.Training.Seed
	}
	return generationPlan{
		numSamples: numSamples, baseSeed: baseSeed, eosTokenID: eosTokenID,
		legacyOutput: numSamples == 1 && opts.GenerationSeed == 0 && opts.OutputPath == "",
	}, nil
}

func generationSeedForSample(base int64, sampleIndex int) int64 {
	if sampleIndex <= 0 {
		return base
	}
	x := uint64(base) ^ uint64(sampleIndex)*generationSeedMix
	x += generationSeedMix
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return int64(x ^ (x >> 31))
}

func runGenerationSamples(
	plan generationPlan,
	vocab *data.NucleotideVocabulary,
	output generationWriter,
	generate func(sampleIndex int, rng *rand.Rand) (generationSample, error),
) error {
	for sampleIndex := 0; sampleIndex < plan.numSamples; sampleIndex++ {
		seed := generationSeedForSample(plan.baseSeed, sampleIndex)
		sample, err := generate(sampleIndex, rand.New(rand.NewSource(seed)))
		if err != nil {
			return fmt.Errorf("generate sample %d: %w", sampleIndex, err)
		}
		if plan.legacyOutput {
			if sample.notice != "" {
				if _, err := fmt.Fprintln(output, sample.notice); err != nil {
					return err
				}
			}
			if _, err := fmt.Fprintf(output, "generated token_ids:%s\n", formatTokenIDs(sample.tokens)); err != nil {
				return err
			}
			if vocab != nil {
				sequence, err := vocab.Decode(sample.tokens)
				if err != nil {
					return fmt.Errorf("decode generated nucleotide sequence: %w", err)
				}
				if _, err := fmt.Fprintf(output, "generated sequence:%s\n", sequence); err != nil {
					return err
				}
			}
			continue
		}
		if vocab != nil {
			sequence, err := vocab.Decode(sample.tokens)
			if err != nil {
				return fmt.Errorf("decode generated nucleotide sequence: %w", err)
			}
			if _, err := fmt.Fprintln(output, sequence); err != nil {
				return err
			}
			continue
		}
		if _, err := fmt.Fprintln(output, formatTokenIDs(sample.tokens)); err != nil {
			return err
		}
	}
	return nil
}

func generateReplaySample(
	cfg *ArchConfig,
	evaluator causalGenerationEvaluator,
	opts GenerateOptions,
	eosTokenID int,
	rng *rand.Rand,
	vocab *data.NucleotideVocabulary,
) (generationSample, error) {
	context, err := generationPromptTokensWithVocabulary(opts.Prompt, cfg.VocabSize, rng, vocab)
	if err != nil {
		return generationSample{}, err
	}
	if len(context) > cfg.SeqLen {
		return generationSample{}, fmt.Errorf("prompt length %d exceeds seq_len %d", len(context), cfg.SeqLen)
	}
	if eosTokenID >= 0 && context[len(context)-1] == eosTokenID {
		return generationSample{tokens: context}, nil
	}
	for step := 0; step < opts.MaxTokens; step++ {
		if len(context) >= cfg.SeqLen {
			return generationSample{tokens: context, notice: fmt.Sprintf("stopped at seq_len limit (%d tokens)", cfg.SeqLen)}, nil
		}
		xTok, yTok, lastPos := generationBatch(context, cfg.SeqLen)
		if _, err := evaluator.EvaluateGPU(xTok, yTok, 1, cfg.SeqLen); err != nil {
			return generationSample{}, fmt.Errorf("generate step %d: %w", step, err)
		}
		logits, err := evaluator.ReadOutput("logits", []int{cfg.SeqLen, cfg.VocabSize})
		if err != nil {
			return generationSample{}, fmt.Errorf("read logits: %w", err)
		}
		next, err := sampleNextToken(logits[lastPos*cfg.VocabSize:(lastPos+1)*cfg.VocabSize], opts.Temperature, opts.TopK, rng)
		if err != nil {
			return generationSample{}, fmt.Errorf("sample token at step %d: %w", step, err)
		}
		context = append(context, next)
		if next == eosTokenID {
			break
		}
	}
	return generationSample{tokens: context}, nil
}

func generateTTTMLPStatefulSample(
	session *TTTMLPInferenceSession,
	cfg *ArchConfig,
	opts GenerateOptions,
	eosTokenID int,
	rng *rand.Rand,
	vocab *data.NucleotideVocabulary,
) (generationSample, error) {
	state, err := session.NewState()
	if err != nil {
		return generationSample{}, err
	}
	result, runErr := func() (generationSample, error) {
		context, err := generationPromptTokensWithVocabulary(opts.Prompt, cfg.VocabSize, rng, vocab)
		if err != nil {
			return generationSample{}, err
		}
		notice := fmt.Sprintf("TTT-MLP stateful inference enabled (prompt_tokens=%d)", len(context))
		if eosTokenID >= 0 && context[len(context)-1] == eosTokenID {
			return generationSample{tokens: context, notice: notice}, nil
		}
		logits, err := session.PrefillLast(state, context)
		if err != nil {
			return generationSample{}, fmt.Errorf("stateful TTT-MLP prefill: %w", err)
		}
		for step := 0; step < opts.MaxTokens; step++ {
			last := logits[len(logits)-cfg.VocabSize:]
			next, err := sampleNextToken(last, opts.Temperature, opts.TopK, rng)
			if err != nil {
				return generationSample{}, fmt.Errorf("sample token at step %d: %w", step, err)
			}
			context = append(context, next)
			if next == eosTokenID || step+1 == opts.MaxTokens {
				break
			}
			logits, err = session.Decode(state, next)
			if err != nil {
				return generationSample{}, fmt.Errorf("stateful TTT-MLP decode step %d: %w", step, err)
			}
		}
		return generationSample{tokens: context, notice: notice}, nil
	}()
	return result, errors.Join(runErr, state.Close())
}
