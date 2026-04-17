package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/train"
)

func main() {
	mode := flag.String("mode", "arch", "run mode: smoke, arch, arch_race, prepare, count, eval, hiddenstats, generate (training configs may set training.target_val_loss for early stopping)")
	configPath := flag.String("config", "", "path to architecture JSON config")
	configsDir := flag.String("configs", "", "directory of JSON configs (for arch_race mode)")
	trainPattern := flag.String("train", "", "glob pattern for training data shards")
	safetensorsPath := flag.String("safetensors", "", "export weights to safetensors file after training")
	safetensorsLoad := flag.String("safetensors-load", "", "load weights from safetensors file before training (resume/eval)")
	quantize := flag.String("quantize", "none", "weight quantization mode: none, int8, int6")
	flagFullEval := flag.Bool("eval", false, "run full validation BPB evaluation after training")
	lutDir := flag.String("lut-dir", "data", "directory containing BPB lookup tables (bytes_per_token.bin, etc.)")
	checkpointDir := flag.String("checkpoint-dir", "", "directory for periodic safetensors checkpoints")
	checkpointEvery := flag.Int("checkpoint-every", 0, "save a safetensors checkpoint every N training steps (0 disables)")
	maxTokens := flag.Int("max-tokens", 256, "maximum number of tokens to generate (generate mode)")
	temperature := flag.Float64("temperature", 0.8, "sampling temperature (generate mode)")
	topK := flag.Int("top-k", 40, "top-k sampling cutoff (generate mode)")
	prompt := flag.String("prompt", "", "prompt for generate mode, e.g. token_ids:0,1,2")

	// prepare mode flags
	prepInput := flag.String("input", "", "input text file, JSONL, or directory (prepare mode)")
	prepOutput := flag.String("output", "", "output directory for shards (prepare mode) or output file (hiddenstats mode)")
	prepVocabSize := flag.Int("vocab-size", 1024, "BPE vocabulary size (prepare mode)")
	prepValSplit := flag.Float64("val-split", 0.1, "fraction of tokens for validation (prepare mode)")
	prepTokenizerPath := flag.String("tokenizer-path", "", "path to pre-trained tokenizer.json (prepare mode)")
	prepTextField := flag.String("text-field", "text", "JSON field for text in JSONL (prepare mode)")

	flag.Parse()

	switch *quantize {
	case "none", "int8", "int6":
		// valid
	default:
		fmt.Fprintf(os.Stderr, "error: unsupported -quantize mode %q (supported: none, int8, int6)\n", *quantize)
		os.Exit(1)
	}

	// smoke, prepare, and count modes handle availability themselves
	if *mode == "smoke" {
		must(train.RunSmoke())
		return
	}
	if *mode == "prepare" {
		must(train.RunPrepare(train.PrepareOptions{
			Input:         *prepInput,
			Output:        *prepOutput,
			VocabSize:     *prepVocabSize,
			ValSplit:      *prepValSplit,
			TokenizerPath: *prepTokenizerPath,
			TextFieldName: *prepTextField,
		}))
		return
	}
	if *mode == "count" {
		must(train.RunCount(*configPath))
		return
	}

	if !train.MLXAvailable() {
		fmt.Fprintln(os.Stderr, "error: MLX backend unavailable\n  rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab\n  requires macOS with Apple Silicon and MLX installed (pip install mlx)")
		os.Exit(1)
	}

	switch *mode {
	case "arch":
		must(train.RunArch(*configPath, *trainPattern, *safetensorsPath, *safetensorsLoad, *quantize, *flagFullEval, *lutDir, *checkpointDir, *checkpointEvery))
	case "arch_race":
		must(train.RunArchRace(*configsDir, *trainPattern, train.TrainOptions{
			SafetensorsPath: *safetensorsPath,
			SafetensorsLoad: *safetensorsLoad,
			Quantize:        *quantize,
			DoFullEval:      *flagFullEval,
			LUTDir:          *lutDir,
			CheckpointDir:   *checkpointDir,
			CheckpointEvery: *checkpointEvery,
		}))
	case "eval":
		must(train.RunEvalMode(*configPath, *trainPattern, *safetensorsLoad))
	case "hiddenstats":
		must(train.RunHiddenstats(*configPath, *trainPattern, *safetensorsLoad, *prepOutput))
	case "generate":
		must(train.RunGenerate(*configPath, *safetensorsLoad, *maxTokens, float32(*temperature), *topK, *prompt))
	default:
		must(fmt.Errorf("unknown mode %q (supported: smoke, arch, arch_race, prepare, count, eval, hiddenstats, generate)", *mode))
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
