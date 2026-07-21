package train

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

var errNoFiniteLogits = errors.New("no finite logits remain after constraints")
var errGrammarNoContinuation = errors.New("grammar has no legal continuation")

// LogitProcessor incrementally constrains one generated sample. Start consumes
// the prompt once, Mask edits the current vocab-wide logit row in place, and
// Accept advances state after the selected token.
type LogitProcessor interface {
	Start(prompt []int) error
	Mask(step int, logits []float32) error
	Accept(token int) error
	Finish() error
}

// LogitProcessorFactory returns a fresh processor for each in-flight sample.
// Implementations may share immutable grammar data through the closure.
type LogitProcessorFactory func() LogitProcessor

func buildGenerationLogitProcessorFactory(opts GenerateOptions, cfg *ArchConfig, vocab *data.NucleotideVocabulary, eosTokenID int) (LogitProcessorFactory, error) {
	configured := 0
	for _, present := range []bool{opts.GrammarTablePath != "", opts.GrammarPath != "", opts.GrammarString != ""} {
		if present {
			configured++
		}
	}
	if configured > 1 {
		return nil, fmt.Errorf("only one of -grammar-table, -grammar, and -grammar-string may be set")
	}
	if configured > 0 && opts.NewLogitProcessor != nil {
		return nil, fmt.Errorf("CLI grammar options cannot be combined with GenerateOptions.NewLogitProcessor")
	}
	if configured == 0 {
		if opts.GrammarPromptMode != "" && normalizeGrammarPromptMode(opts.GrammarPromptMode) != grammarPromptConsume {
			return nil, fmt.Errorf("-grammar-prompt-mode requires a grammar option")
		}
		return opts.NewLogitProcessor, nil
	}
	mode := normalizeGrammarPromptMode(opts.GrammarPromptMode)
	if mode == "" {
		return nil, fmt.Errorf("-grammar-prompt-mode=%q must be consume or ignore", opts.GrammarPromptMode)
	}
	var factory LogitProcessorFactory
	if opts.GrammarTablePath != "" {
		table, err := loadTokenDFA(opts.GrammarTablePath, cfg.VocabSize, eosTokenID)
		if err != nil {
			return nil, err
		}
		factory = func() LogitProcessor { return &tokenDFAProcessor{dfa: table} }
	} else {
		var source string
		if opts.GrammarPath != "" {
			body, err := os.ReadFile(opts.GrammarPath)
			if err != nil {
				return nil, fmt.Errorf("read grammar %q: %w", opts.GrammarPath, err)
			}
			source = string(body)
		} else {
			source = opts.GrammarString
		}
		var err error
		factory, err = buildGBNFLogitProcessorFactory(source, opts.TokenizerPath, opts.ConfigPath, opts.SafetensorsLoad, cfg, vocab, eosTokenID)
		if err != nil {
			return nil, err
		}
	}
	if mode == grammarPromptIgnore {
		base := factory
		factory = func() LogitProcessor { return &promptIgnoringLogitProcessor{inner: base()} }
	}
	return factory, nil
}

const (
	grammarPromptConsume = "consume"
	grammarPromptIgnore  = "ignore"
)

func normalizeGrammarPromptMode(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return grammarPromptConsume
	}
	switch value {
	case grammarPromptConsume, grammarPromptIgnore:
		return value
	default:
		return ""
	}
}

type promptIgnoringLogitProcessor struct {
	inner LogitProcessor
}

func (p *promptIgnoringLogitProcessor) Start(_ []int) error {
	if p == nil || p.inner == nil {
		return fmt.Errorf("logit processor is not initialized")
	}
	return p.inner.Start(nil)
}

func (p *promptIgnoringLogitProcessor) Mask(step int, logits []float32) error {
	return p.inner.Mask(step, logits)
}

func (p *promptIgnoringLogitProcessor) Accept(token int) error {
	return p.inner.Accept(token)
}

func (p *promptIgnoringLogitProcessor) Finish() error {
	return p.inner.Finish()
}

func startLogitProcessor(factory LogitProcessorFactory, prompt []int) (LogitProcessor, error) {
	if factory == nil {
		return nil, nil
	}
	processor := factory()
	if processor == nil {
		return nil, fmt.Errorf("logit processor factory returned nil")
	}
	if err := processor.Start(prompt); err != nil {
		return nil, fmt.Errorf("initialize logit processor from prompt: %w", err)
	}
	return processor, nil
}

func constrainAndSampleNextToken(processor LogitProcessor, step int, logits []float32, temperature float32, topK int, rng *rand.Rand) (int, error) {
	if processor != nil {
		if err := processor.Mask(step, logits); err != nil {
			if errors.Is(err, errGrammarNoContinuation) {
				return 0, fmt.Errorf("grammar permits no continuation at step %d: %w", step, err)
			}
			return 0, fmt.Errorf("apply logit constraints: %w", err)
		}
	}
	next, err := sampleNextToken(logits, temperature, topK, rng)
	if err != nil {
		if processor != nil && errors.Is(err, errNoFiniteLogits) {
			return 0, fmt.Errorf("grammar permits no continuation at step %d: %w", step, err)
		}
		return 0, err
	}
	if processor != nil {
		if err := processor.Accept(next); err != nil {
			return 0, fmt.Errorf("advance logit constraints with token %d: %w", next, err)
		}
	}
	return next, nil
}

func finishLogitProcessor(processor LogitProcessor) error {
	if processor == nil {
		return nil
	}
	if err := processor.Finish(); err != nil {
		return fmt.Errorf("generated output does not complete the grammar: %w", err)
	}
	return nil
}
