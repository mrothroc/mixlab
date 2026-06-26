package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"

	"github.com/mrothroc/mixlab/logits"
	"github.com/mrothroc/mixlab/train"
)

func main() {
	mode := flag.String("mode", "arch", "run mode: smoke, arch, arch_race, prepare, count, eval, hiddenstats, generate, generate-diffusion, export-hf, parity (training configs may set training.target_val_loss for early stopping)")
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
	prompt := flag.String("prompt", "", "prompt for generate mode, e.g. token_ids:0,1,2")
	logprobsOut := flag.String("logprobs-out", "", "write per-token eval NLLs to a binary file (eval mode)")
	ranksOut := flag.String("ranks-out", "", "write per-token target ranks to a binary file (eval mode); can be combined with -logprobs-out for a single eval pass")
	uncertaintyOut := flag.String("uncertainty-out", "", "write per-token uncertainty metrics (top-1 prob, entropy, margin) to a binary file (eval mode); can be combined with -logprobs-out and -ranks-out for a single eval pass")
	logitsOut := flag.String("logits-out", "", "write per-token full-vocab logits to a binary file (eval mode); can be combined with -logprobs-out, -ranks-out, -uncertainty-out for a single eval pass")
	logitsDType := flag.String("logits-dtype", "float16", "on-disk dtype for -logits-out: float16 (default) or float32")
	logitsForm := flag.String("logits-form", "raw", "encoding for -logits-out rows: raw (default) or logprobs (log_softmax)")
	hfDir := flag.String("hf", "", "Hugging Face export directory to compare against native inference (parity mode)")
	parityThreshold := flag.Float64("threshold", 0.05, "maximum allowed native-vs-HF mean NLL difference (parity mode)")
	maxLogitDiff := flag.Float64("max-logit-diff", 1e-3, "maximum allowed native-vs-HF absolute logit difference on the sampled rows; <=0 disables (parity mode)")
	parityLogitTokens := flag.Int("parity-logit-tokens", 0, "number of token pairs to sample for logit comparison; rounded up to full eval batches, 0 uses one batch (parity mode)")
	parityPython := flag.String("parity-python", "", "Python interpreter for the HF parity checker; defaults to HF_PARITY_PYTHON or python3 (parity mode)")

	// profiling flags
	cpuProfile := flag.String("cpuprofile", "", "write CPU profile to file")
	memProfile := flag.String("memprofile", "", "write memory profile to file")

	// prepare mode flags
	prepInput := flag.String("input", "", "input text file, JSONL, or directory (prepare mode)")
	prepOutput := flag.String("output", "", "output directory for shards (prepare mode), Hugging Face directory (export-hf mode), or output file (hiddenstats mode)")
	prepVocabSize := flag.Int("vocab-size", 1024, "BPE vocabulary size (prepare mode)")
	prepValSplit := flag.Float64("val-split", 0.1, "fraction of tokens for validation (prepare mode)")
	prepTokenizerPath := flag.String("tokenizer-path", "", "path to pre-trained tokenizer.json (prepare mode)")
	prepTextField := flag.String("text-field", "text", "JSON field for text in JSONL (prepare mode)")
	prepCharVocabSize := flag.Int("char-vocab-size", 0, "write tokenizer-level char_features.bin with this char vocab size; 0 disables (prepare mode)")
	prepCharMaxPerToken := flag.Int("char-max-per-token", 16, "fixed char feature slots per token when -char-vocab-size is enabled (prepare mode)")

	flag.Parse()
	providedFlags := providedFlagSet()

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
		must(train.RunPrepare(train.PrepareOptions{
			Input:           *prepInput,
			Output:          *prepOutput,
			VocabSize:       *prepVocabSize,
			ValSplit:        *prepValSplit,
			TokenizerPath:   *prepTokenizerPath,
			TextFieldName:   *prepTextField,
			CharVocabSize:   *prepCharVocabSize,
			CharMaxPerToken: *prepCharMaxPerToken,
		}))
		return
	}
	if *mode == "count" {
		must(train.RunCount(*configPath))
		return
	}
	if *mode == "export-hf" {
		must(train.RunExportHF(train.ExportHFOptions{
			ConfigPath:      *configPath,
			SafetensorsLoad: *safetensorsLoad,
			OutputDir:       *prepOutput,
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
		DoFullEval:      *flagFullEval,
		LUTDir:          *lutDir,
		CheckpointDir:   *checkpointDir,
		CheckpointEvery: *checkpointEvery,
		LogEvery:        *logEvery,
		ValEvery:        *valEvery,
		Timing:          *timing,
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
		must(train.RunHiddenstats(*configPath, *trainPattern, *safetensorsLoad, *prepOutput))
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
	case "parity":
		must(train.RunParity(train.ParityOptions{
			ConfigPath:      *configPath,
			SafetensorsLoad: *safetensorsLoad,
			HFDir:           *hfDir,
			TokenPattern:    *trainPattern,
			Python:          *parityPython,
			LossThreshold:   *parityThreshold,
			MaxLogitDiff:    *maxLogitDiff,
			LogitTokens:     *parityLogitTokens,
		}))
	default:
		must(fmt.Errorf("unknown mode %q (supported: smoke, arch, arch_race, prepare, count, eval, hiddenstats, generate, generate-diffusion, export-hf, parity)", *mode))
	}
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
