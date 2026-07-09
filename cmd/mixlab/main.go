package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"

	"github.com/mrothroc/mixlab/logits"
	"github.com/mrothroc/mixlab/train"
)

func main() {
	mode := flag.String("mode", "arch", "run mode: smoke, arch, arch_race, prepare, prepare-pairs, count, eval, hiddenstats, generate, generate-diffusion, score-diffusion, score-electra, score-ebm, export-hf, parity (training configs may set training.target_val_loss for early stopping)")
	configPath := flag.String("config", "", "path to architecture JSON config")
	configsDir := flag.String("configs", "", "directory of JSON configs (for arch_race mode)")
	trainPattern := flag.String("train", "", "glob pattern for training data shards")
	safetensorsPath := flag.String("safetensors", "", "export weights to safetensors file after training")
	safetensorsLoad := flag.String("safetensors-load", "", "load weights from safetensors file before training (resume/eval)")
	quantize := flag.String("quantize", "none", "weight quantization mode: none, int8, int6")
	quantMethod := flag.String("quant-method", "quantile", `quantization clipping method: "quantile" or "sdclip"`)
	quantK := flag.Float64("quant-k", 12.85, "SDClip k for matrix weights")
	quantKEmbed := flag.Float64("quant-k-embed", 20.0, "SDClip k for embedding weights")
	flagFullEval := flag.Bool("eval", false, "run full validation BPB evaluation after training")
	lutDir := flag.String("lut-dir", "data", "directory containing BPB lookup tables (bytes_per_token.bin, etc.)")
	checkpointDir := flag.String("checkpoint-dir", "", "directory for periodic safetensors checkpoints")
	checkpointEvery := flag.Int("checkpoint-every", 0, "save a safetensors checkpoint every N training steps (0 disables)")
	logEvery := flag.Int("log-every", 0, "print progress every N training steps (0 uses default; MIXLAB_LOG_EVERY overrides)")
	valEvery := flag.Int("val-every", 0, "run validation every N training steps (0 uses default; MIXLAB_VAL_EVERY overrides)")
	evalAfterTrain := flag.Bool("eval-after-train", false, "run full validation BPB evaluation after training; clearer alias for -eval")
	swaStart := flag.Int("swa-start", 0, "override training.swa_start: step at which SWA/EMA accumulation starts; 0 disables")
	swaDecay := flag.Float64("swa-decay", 0.999, "override training.swa_decay: EMA decay for averaged weights")
	swaInterval := flag.Int("swa-interval", 10, "override training.swa_interval: update cadence for SWA/EMA accumulation")
	timing := flag.Bool("timing", false, "print per-step timing breakdown")
	maxTokens := flag.Int("max-tokens", 256, "maximum number of tokens to generate (generate mode)")
	temperature := flag.Float64("temperature", 0.8, "sampling temperature (generate mode)")
	topK := flag.Int("top-k", 40, "top-k sampling cutoff (generate mode)")
	diffusionStepsPerBlock := flag.Int("diffusion-steps-per-block", 0, "override training.diffusion.steps_per_block for generate-diffusion (0 uses config)")
	diffusionConfidenceThreshold := flag.Float64("diffusion-confidence-threshold", 0, "override training.diffusion.confidence_threshold for generate-diffusion when explicitly set")
	diffusionCommitFloor := flag.Int("diffusion-commit-floor", 0, "override training.diffusion.commit_floor for generate-diffusion (0 uses config)")
	diffusionTemperature := flag.Float64("diffusion-temperature", 0, "diffusion sampling temperature; 0 keeps deterministic argmax")
	diffusionTopK := flag.Int("diffusion-top-k", 0, "diffusion top-k sampling cutoff when -diffusion-temperature > 0; 0 disables cutoff")
	diffusionTraceOut := flag.String("diffusion-trace-out", "", "write generate-diffusion sampler telemetry as JSONL")
	scoreIn := flag.String("score-in", "", "JSONL token-id sequences or pairs to score (score modes)")
	scoreOut := flag.String("score-out", "", "JSONL output path for score mode results")
	scoreMode := flag.String("score-mode", "block_causal", "diffusion scoring mode: block_causal")
	scoreSkipFirst := flag.Int("score-skip-first", 0, "globally skip the first N tokens when scoring JSONL token sequences")
	scorePositionBatch := flag.Int("score-position-batch", 0, "masked positions per diffusion or score-ebm full-sequence PLL scoring forward; <=0 auto-selects a memory-bounded value")
	scoreBatch := flag.Int("score-batch", 0, "sequence rows per ELECTRA detector or score-ebm non-full-seq scoring forward; <=0 uses a conservative default")
	scoreEmitTokenEnergy := flag.Bool("score-emit-token-energy", false, "include per-token energy values in score-ebm output for differing-span native energy configs")
	scorePLLAggregation := flag.String("score-pll-aggregation", "config", "score-ebm scorer PLL aggregation: config, full_seq, differing_span, or dependent_window")
	scorePLLWindow := flag.Int("score-pll-window", 0, "context window K for score-ebm -score-pll-aggregation=dependent_window")
	scorePLLAttributionDump := flag.String("score-pll-attribution-dump", "", "write score-ebm PLL per-token attribution JSONL for pair records")
	scorePLLSkipTokenIDs := flag.String("score-pll-skip-token-ids", "", "comma-separated token IDs to skip for score-ebm PLL; mlm_mask_token_id is always skipped")
	prompt := flag.String("prompt", "", "prompt for generate mode, e.g. token_ids:0,1,2")
	logprobsOut := flag.String("logprobs-out", "", "write per-token eval NLLs to a binary file (eval mode)")
	ranksOut := flag.String("ranks-out", "", "write per-token target ranks to a binary file (eval mode); can be combined with -logprobs-out for a single eval pass")
	uncertaintyOut := flag.String("uncertainty-out", "", "write per-token uncertainty metrics (top-1 prob, entropy, margin) to a binary file (eval mode); can be combined with -logprobs-out and -ranks-out for a single eval pass")
	logitsOut := flag.String("logits-out", "", "write per-token full-vocab logits to a binary file (eval mode); can be combined with -logprobs-out, -ranks-out, -uncertainty-out for a single eval pass")
	logitsDType := flag.String("logits-dtype", "float16", "on-disk dtype for -logits-out: float16 (default) or float32")
	logitsForm := flag.String("logits-form", "raw", "encoding for -logits-out rows: raw (default) or logprobs (log_softmax)")
	hfDir := flag.String("hf", "", "Hugging Face export directory to compare against native inference (parity mode)")
	parityThreshold := flag.Float64("threshold", 0.05, "maximum allowed native-vs-HF mean NLL difference (parity mode)")
	parityLossThreshold := flag.Float64("parity-loss-threshold", 0.05, "maximum allowed native-vs-HF mean NLL difference; clearer alias for -threshold")
	maxLogitDiff := flag.Float64("max-logit-diff", 1e-3, "maximum allowed native-vs-HF absolute logit difference on the sampled rows; <=0 disables (parity mode)")
	parityLogitTokens := flag.Int("parity-logit-tokens", 0, "number of token pairs to sample for logit comparison; rounded up to full eval batches, 0 uses one batch (parity mode)")
	parityPython := flag.String("parity-python", "", "Python interpreter for the HF parity checker; defaults to HF_PARITY_PYTHON or python3 (parity mode)")

	// profiling flags
	cpuProfile := flag.String("cpuprofile", "", "write CPU profile to file")
	memProfile := flag.String("memprofile", "", "write memory profile to file")
	pprofAddr := flag.String("pprof-addr", "", "serve live pprof and Mixlab telemetry HTTP endpoints on this address, e.g. 127.0.0.1:6060")
	telemetryOut := flag.String("telemetry-out", "", "write periodic Mixlab telemetry snapshots as JSONL")

	// prepare mode flags
	prepInput := flag.String("input", "", "input text file, JSONL, or directory (prepare mode)")
	prepOutput := flag.String("output", "", "legacy output path for prepare, export-hf, or hiddenstats; prefer mode-specific output aliases in new scripts")
	prepareOutputDir := flag.String("prepare-output-dir", "", "output directory for shards; clearer alias for -output in prepare mode")
	exportDir := flag.String("export-dir", "", "Hugging Face output directory; clearer alias for -output in export-hf mode")
	hiddenstatsOut := flag.String("hiddenstats-out", "", "output file for hiddenstats mode; clearer alias for -output")
	prepVocabSize := flag.Int("vocab-size", 1024, "BPE vocabulary size (prepare mode)")
	prepValSplit := flag.Float64("val-split", 0.1, "fraction of tokens for validation (prepare mode)")
	prepTokenizerPath := flag.String("tokenizer-path", "", "path to tokenizer.json for prepare reuse or export-hf bundling")
	prepTextField := flag.String("text-field", "text", "JSON field for text in JSONL (prepare mode)")
	prepCharVocabSize := flag.Int("char-vocab-size", 0, "write tokenizer-level char_features.bin with this char vocab size; 0 disables (prepare mode)")
	prepCharMaxPerToken := flag.Int("char-max-per-token", 16, "fixed char feature slots per token when -char-vocab-size is enabled (prepare mode)")
	prepMinimalPairOut := flag.String("minimal-pair-out", "", "write corpus-derived minimal-pair JSONL to this path (prepare mode)")
	prepMinimalPairCorruptions := flag.String("minimal-pair-corruptions", "agreement,attractor,word_order", "comma-separated minimal-pair corruption families (prepare mode)")
	prepMinimalPairWeights := flag.String("minimal-pair-weights", "", "minimal-pair family weights as JSON object or family=value list (prepare mode)")
	prepMinimalPairMorphology := flag.String("minimal-pair-morphology", "induced", "minimal-pair morphology source: induced (prepare mode)")
	prepMinimalPairMaxPairs := flag.Int("minimal-pair-max-pairs", 0, "maximum generated minimal pairs; 0 lets prepare choose from input size (prepare mode)")
	prepMinimalPairSeed := flag.Int("minimal-pair-seed", 1234, "deterministic seed for minimal-pair generation (prepare mode)")
	prepMinimalPairReportOut := flag.String("minimal-pair-report-out", "", "write minimal-pair generation report JSON (prepare mode)")
	prepMinimalPairSampleOut := flag.String("minimal-pair-sample-out", "", "write auditable minimal-pair sample JSONL (prepare mode)")
	prepMinimalPairSampleCount := flag.Int("minimal-pair-sample-count", 20, "maximum minimal-pair audit samples to write (prepare mode)")
	pairIn := flag.String("pair-in", "", "minimal-pair JSONL input for prepare-pairs mode")
	pairOut := flag.String("pair-out", "", "compiled minimal-pair binary output for prepare-pairs mode; omit to validate only")
	pairMaxLen := flag.Int("pair-max-len", 0, "maximum clean/corrupt token length for prepare-pairs; 0 uses config seq_len when -config is provided")

	flag.Usage = func() {
		printUsage(os.Stderr, requestedHelpMode(os.Args[1:]))
	}
	flag.Parse()
	providedFlags := providedFlagSet()
	effectiveParityThreshold := *parityThreshold
	if providedFlags["parity-loss-threshold"] {
		effectiveParityThreshold = *parityLossThreshold
	}

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: create CPU profile: %v\n", err)
			os.Exit(1)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			_ = f.Close()
			fmt.Fprintf(os.Stderr, "error: start CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer func() {
			pprof.StopCPUProfile()
			_ = f.Close()
		}()
	}
	if *memProfile != "" {
		defer func() {
			f, err := os.Create(*memProfile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: create mem profile: %v\n", err)
				return
			}
			runtime.GC()
			_ = pprof.WriteHeapProfile(f)
			_ = f.Close()
		}()
	}

	// Configure backend graph limits before MLX initializes. User-provided
	// environment overrides are preserved.
	train.ConfigureCUDAGraphLimits(*configPath, *configsDir)

	switch *quantize {
	case "none", "int8", "int6":
		// valid
	default:
		fmt.Fprintf(os.Stderr, "error: unsupported -quantize mode %q (supported: none, int8, int6)\n", *quantize)
		os.Exit(1)
	}
	switch *quantMethod {
	case "", "quantile", "sdclip":
		// valid
	default:
		fmt.Fprintf(os.Stderr, "error: unsupported -quant-method %q (supported: quantile, sdclip)\n", *quantMethod)
		os.Exit(1)
	}

	// smoke, prepare, and count modes handle availability themselves
	if *mode == "smoke" {
		must(train.RunSmoke())
		return
	}
	if *mode == "prepare" {
		prepareOutput, err := aliasedStringFlagValue(*prepOutput, "prepare-output-dir", *prepareOutputDir, providedFlags)
		must(err)
		must(train.RunPrepare(train.PrepareOptions{
			Input:                  *prepInput,
			Output:                 prepareOutput,
			VocabSize:              *prepVocabSize,
			ValSplit:               *prepValSplit,
			TokenizerPath:          *prepTokenizerPath,
			TextFieldName:          *prepTextField,
			CharVocabSize:          *prepCharVocabSize,
			CharMaxPerToken:        *prepCharMaxPerToken,
			MinimalPairOut:         *prepMinimalPairOut,
			MinimalPairCorruptions: *prepMinimalPairCorruptions,
			MinimalPairWeights:     *prepMinimalPairWeights,
			MinimalPairMorphology:  *prepMinimalPairMorphology,
			MinimalPairMaxPairs:    *prepMinimalPairMaxPairs,
			MinimalPairSeed:        *prepMinimalPairSeed,
			MinimalPairReportOut:   *prepMinimalPairReportOut,
			MinimalPairSampleOut:   *prepMinimalPairSampleOut,
			MinimalPairSampleCount: *prepMinimalPairSampleCount,
		}))
		return
	}
	if *mode == "prepare-pairs" {
		must(train.RunPreparePairsWithOptions(train.PreparePairsOptions{
			ConfigPath: *configPath,
			PairIn:     *pairIn,
			PairOut:    *pairOut,
			VocabSize:  *prepVocabSize,
			MaxLen:     *pairMaxLen,
		}))
		return
	}
	if *mode == "count" {
		must(train.RunCount(*configPath))
		return
	}
	if *mode == "export-hf" {
		exportOutput, err := aliasedStringFlagValue(*prepOutput, "export-dir", *exportDir, providedFlags)
		must(err)
		must(train.RunExportHF(train.ExportHFOptions{
			ConfigPath:      *configPath,
			SafetensorsLoad: *safetensorsLoad,
			OutputDir:       exportOutput,
			TokenizerSource: *prepTokenizerPath,
		}))
		return
	}

	if !train.MLXAvailable() {
		fmt.Fprintln(os.Stderr, "error: MLX backend unavailable\n  rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab\n  macOS: requires Apple Silicon and MLX (brew install mlx)\n  Linux: requires CUDA and MLX built from source (see docker/README.md)")
		os.Exit(1)
	}

	opts := train.TrainOptions{
		SafetensorsPath: *safetensorsPath,
		SafetensorsLoad: *safetensorsLoad,
		Quantize:        *quantize,
		QuantMethod:     *quantMethod,
		QuantK:          float32(*quantK),
		QuantKEmbed:     float32(*quantKEmbed),
		DoFullEval:      *flagFullEval || *evalAfterTrain,
		LUTDir:          *lutDir,
		CheckpointDir:   *checkpointDir,
		CheckpointEvery: *checkpointEvery,
		LogEvery:        *logEvery,
		ValEvery:        *valEvery,
		Timing:          *timing,
		PProfAddr:       *pprofAddr,
		TelemetryOut:    *telemetryOut,
	}
	if providedFlags["swa-start"] {
		v := *swaStart
		opts.SWAStartOverride = &v
	}
	if providedFlags["swa-decay"] {
		v := float32(*swaDecay)
		opts.SWADecayOverride = &v
	}
	if providedFlags["swa-interval"] {
		v := *swaInterval
		opts.SWAIntervalOverride = &v
	}

	switch *mode {
	case "arch":
		must(train.RunArch(*configPath, *trainPattern, opts))
	case "arch_race":
		must(train.RunArchRace(*configsDir, *trainPattern, opts))
	case "eval":
		switch {
		case *logprobsOut != "" || *ranksOut != "" || *uncertaintyOut != "" || *logitsOut != "":
			dtype, err := logits.ParseDType(*logitsDType)
			if err != nil {
				must(err)
			}
			form, err := logits.ParseForm(*logitsForm)
			if err != nil {
				must(err)
			}
			must(train.RunEvalExports(*configPath, *trainPattern, *safetensorsLoad, *lutDir, train.EvalExportOptions{
				LogprobsOut:    *logprobsOut,
				RanksOut:       *ranksOut,
				UncertaintyOut: *uncertaintyOut,
				LogitsOut:      *logitsOut,
				LogitsDType:    dtype,
				LogitsForm:     form,
			}))
		default:
			must(train.RunEvalModeWithLUT(*configPath, *trainPattern, *safetensorsLoad, *lutDir))
		}
	case "hiddenstats":
		hiddenstatsOutput, err := aliasedStringFlagValue(*prepOutput, "hiddenstats-out", *hiddenstatsOut, providedFlags)
		must(err)
		must(train.RunHiddenstats(*configPath, *trainPattern, *safetensorsLoad, hiddenstatsOutput))
	case "generate":
		must(train.RunGenerate(*configPath, *safetensorsLoad, *maxTokens, float32(*temperature), *topK, *prompt))
	case "generate-diffusion":
		var confidenceOverride *float64
		if providedFlags["diffusion-confidence-threshold"] {
			v := *diffusionConfidenceThreshold
			confidenceOverride = &v
		}
		must(train.RunGenerateDiffusionWithOptions(train.GenerateDiffusionOptions{
			ConfigPath:                   *configPath,
			SafetensorsLoad:              *safetensorsLoad,
			MaxTokens:                    *maxTokens,
			Prompt:                       *prompt,
			DiffusionStepsPerBlock:       *diffusionStepsPerBlock,
			DiffusionConfidenceThreshold: confidenceOverride,
			DiffusionCommitFloor:         *diffusionCommitFloor,
			DiffusionTemperature:         float32(*diffusionTemperature),
			DiffusionTopK:                *diffusionTopK,
			DiffusionTraceOut:            *diffusionTraceOut,
		}))
	case "score-diffusion":
		must(train.RunScoreDiffusionWithOptions(train.ScoreDiffusionOptions{
			ConfigPath:         *configPath,
			SafetensorsLoad:    *safetensorsLoad,
			ScoreIn:            *scoreIn,
			ScoreOut:           *scoreOut,
			ScoreMode:          *scoreMode,
			ScoreSkipFirst:     *scoreSkipFirst,
			ScorePositionBatch: *scorePositionBatch,
		}))
	case "score-electra":
		must(train.RunScoreElectraWithOptions(train.ScoreElectraOptions{
			ConfigPath:      *configPath,
			SafetensorsLoad: *safetensorsLoad,
			ScoreIn:         *scoreIn,
			ScoreOut:        *scoreOut,
			ScoreSkipFirst:  *scoreSkipFirst,
			ScoreBatch:      *scoreBatch,
		}))
	case "score-ebm":
		must(train.RunScoreEBMWithOptions(train.ScoreEBMOptions{
			ConfigPath:         *configPath,
			SafetensorsLoad:    *safetensorsLoad,
			ScoreIn:            *scoreIn,
			ScoreOut:           *scoreOut,
			ScoreBatch:         *scoreBatch,
			ScorePositionBatch: *scorePositionBatch,
			PLLAggregation:     *scorePLLAggregation,
			PLLWindow:          *scorePLLWindow,
			PLLSkipTokenIDs:    *scorePLLSkipTokenIDs,
			PLLAttributionDump: *scorePLLAttributionDump,
			EmitTokenEnergy:    *scoreEmitTokenEnergy,
		}))
	case "parity":
		must(train.RunParity(train.ParityOptions{
			ConfigPath:      *configPath,
			SafetensorsLoad: *safetensorsLoad,
			HFDir:           *hfDir,
			TokenPattern:    *trainPattern,
			Python:          *parityPython,
			LossThreshold:   effectiveParityThreshold,
			MaxLogitDiff:    *maxLogitDiff,
			LogitTokens:     *parityLogitTokens,
		}))
	default:
		must(fmt.Errorf("unknown mode %q (supported: smoke, arch, arch_race, prepare, prepare-pairs, count, eval, hiddenstats, generate, generate-diffusion, score-diffusion, score-electra, score-ebm, export-hf, parity)", *mode))
	}
}

type flagGroup struct {
	Title string
	Names []string
}

var supportedModes = []string{"smoke", "arch", "arch_race", "prepare", "prepare-pairs", "count", "eval", "hiddenstats", "generate", "generate-diffusion", "score-diffusion", "score-electra", "score-ebm", "export-hf", "parity"}

var modeFlagGroups = map[string][]flagGroup{
	"arch": {
		{"Required", []string{"config", "train"}},
		{"Checkpointing", []string{"safetensors", "safetensors-load", "checkpoint-dir", "checkpoint-every"}},
		{"Training run controls", []string{"eval", "eval-after-train", "lut-dir", "log-every", "val-every", "timing", "swa-start", "swa-decay", "swa-interval"}},
		{"Quantization", []string{"quantize", "quant-method", "quant-k", "quant-k-embed"}},
		{"Profiling and telemetry", []string{"cpuprofile", "memprofile", "pprof-addr", "telemetry-out"}},
	},
	"arch_race": {
		{"Required", []string{"configs", "train"}},
		{"Checkpointing and eval", []string{"safetensors", "safetensors-load", "eval", "eval-after-train", "lut-dir"}},
		{"Run controls", []string{"log-every", "val-every", "timing"}},
		{"Quantization", []string{"quantize", "quant-method", "quant-k", "quant-k-embed"}},
		{"Profiling and telemetry", []string{"cpuprofile", "memprofile", "pprof-addr", "telemetry-out"}},
	},
	"prepare": {
		{"Required", []string{"input"}},
		{"Output", []string{"prepare-output-dir", "output"}},
		{"Tokenizer/data", []string{"vocab-size", "val-split", "tokenizer-path", "text-field"}},
		{"Character feature artifact", []string{"char-vocab-size", "char-max-per-token"}},
		{"Minimal pair artifact", []string{"minimal-pair-out", "minimal-pair-corruptions", "minimal-pair-weights", "minimal-pair-morphology", "minimal-pair-max-pairs", "minimal-pair-seed", "minimal-pair-report-out", "minimal-pair-sample-out", "minimal-pair-sample-count"}},
	},
	"prepare-pairs": {
		{"Input", []string{"pair-in", "config"}},
		{"Validation", []string{"vocab-size", "pair-max-len"}},
		{"Output", []string{"pair-out"}},
	},
	"count": {
		{"Required", []string{"config"}},
	},
	"eval": {
		{"Required", []string{"config", "train", "safetensors-load"}},
		{"Lookup tables", []string{"lut-dir"}},
		{"Per-token exports", []string{"logprobs-out", "ranks-out", "uncertainty-out", "logits-out", "logits-dtype", "logits-form"}},
	},
	"hiddenstats": {
		{"Required", []string{"config", "train", "safetensors-load"}},
		{"Output", []string{"hiddenstats-out", "output"}},
	},
	"generate": {
		{"Required", []string{"config", "safetensors-load", "prompt"}},
		{"Sampling", []string{"max-tokens", "temperature", "top-k"}},
	},
	"generate-diffusion": {
		{"Required", []string{"config", "safetensors-load", "prompt"}},
		{"Sampling", []string{"max-tokens", "diffusion-steps-per-block", "diffusion-confidence-threshold", "diffusion-commit-floor", "diffusion-temperature", "diffusion-top-k", "diffusion-trace-out"}},
	},
	"score-diffusion": {
		{"Required", []string{"config", "safetensors-load", "score-in", "score-out"}},
		{"Scoring", []string{"score-mode", "score-skip-first", "score-position-batch"}},
	},
	"score-electra": {
		{"Required", []string{"config", "safetensors-load", "score-in", "score-out"}},
		{"Scoring", []string{"score-skip-first", "score-batch"}},
	},
	"score-ebm": {
		{"Required", []string{"config", "safetensors-load", "score-in", "score-out"}},
		{"Scoring", []string{"score-batch", "score-position-batch", "score-pll-aggregation", "score-pll-window", "score-pll-attribution-dump", "score-pll-skip-token-ids", "score-emit-token-energy"}},
	},
	"export-hf": {
		{"Required", []string{"config", "safetensors-load"}},
		{"Output", []string{"export-dir", "output"}},
		{"Tokenizer", []string{"tokenizer-path"}},
	},
	"parity": {
		{"Required", []string{"config", "safetensors-load", "hf", "train"}},
		{"Thresholds", []string{"threshold", "parity-loss-threshold", "max-logit-diff", "parity-logit-tokens"}},
		{"Python", []string{"parity-python"}},
	},
	"smoke": {
		{"Options", []string{}},
	},
}

func requestedHelpMode(args []string) string {
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "-mode" || arg == "--mode":
			if i+1 < len(args) {
				return args[i+1]
			}
		case strings.HasPrefix(arg, "-mode="):
			return strings.TrimPrefix(arg, "-mode=")
		case strings.HasPrefix(arg, "--mode="):
			return strings.TrimPrefix(arg, "--mode=")
		}
	}
	return ""
}

// fprintf/fprintln write usage text to an arbitrary io.Writer (os.Stderr in
// production, a buffer in tests). Usage output has nowhere useful to report a
// write error, so the result is intentionally discarded.
func fprintf(w io.Writer, format string, a ...any) { _, _ = fmt.Fprintf(w, format, a...) }
func fprintln(w io.Writer, a ...any)               { _, _ = fmt.Fprintln(w, a...) }

func printUsage(w io.Writer, mode string) {
	if mode == "" {
		fprintf(w, "Usage: mixlab -mode MODE [flags]\n\n")
		fprintf(w, "Modes: %s\n\n", strings.Join(supportedModes, ", "))
		fprintln(w, "Use `mixlab -mode MODE -h` for mode-specific flags.")
		fprintln(w, "Common flags:")
		printFlagGroup(w, flagGroup{"", []string{"mode", "config", "train", "safetensors", "safetensors-load"}})
		return
	}
	groups, ok := modeFlagGroups[mode]
	if !ok {
		fprintf(w, "Unknown mode %q.\n\n", mode)
		printUsage(w, "")
		return
	}
	fprintf(w, "Usage: mixlab -mode %s [flags]\n\n", mode)
	for _, group := range groups {
		printFlagGroup(w, group)
	}
	if mode == "prepare" || mode == "export-hf" || mode == "hiddenstats" {
		fprintln(w, "\n`-output` remains supported. Prefer the mode-specific alias in new scripts when available.")
	}
}

func printFlagGroup(w io.Writer, group flagGroup) {
	if group.Title != "" {
		fprintf(w, "%s:\n", group.Title)
	}
	if len(group.Names) == 0 {
		fprintln(w, "  (no flags)")
		return
	}
	for _, name := range group.Names {
		f := flag.Lookup(name)
		if f == nil {
			continue
		}
		valueName, usage := flag.UnquoteUsage(f)
		if valueName != "" {
			fprintf(w, "  -%s %s\n", f.Name, valueName)
		} else {
			fprintf(w, "  -%s\n", f.Name)
		}
		fprintf(w, "      %s", usage)
		if f.DefValue != "" && f.DefValue != "false" {
			fprintf(w, " (default %s)", f.DefValue)
		}
		fprintln(w)
	}
}

// aliasedStringFlagValue resolves the legacy -output flag against a
// mode-specific alias (e.g. -export-dir). The alias wins when set; providing
// both with conflicting non-empty values is an error.
func aliasedStringFlagValue(outputValue, aliasName, aliasValue string, provided map[string]bool) (string, error) {
	if !provided[aliasName] {
		return outputValue, nil
	}
	if provided["output"] && outputValue != "" && aliasValue != "" && outputValue != aliasValue {
		return "", fmt.Errorf("-output and -%s both provided with different values", aliasName)
	}
	return aliasValue, nil
}

func providedFlagSet() map[string]bool {
	provided := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) {
		provided[f.Name] = true
	})
	return provided
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
