//go:build mlx && cgo && (darwin || linux)

package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestConstrainedGenerationMLXReplayAndBatch(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := &ArchConfig{
		Name: "constrained_generation_mlx", ModelDim: 8, VocabSize: 5, SeqLen: 4,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.BatchTokens = cfg.SeqLen
	dir, configPath, weightsPath := newBulkGenerationFixture(t, cfg)

	tablePath := filepath.Join(dir, "grammar.json")
	table := `{
  "format":"mixlab.token_dfa","version":1,"vocab_size":5,
  "start_state":"start","eos_token_ids":[1],
  "states":[
    {"name":"start","transitions":{"0":"value"}},
    {"name":"value","transitions":{"2":"complete"}},
    {"name":"complete","transitions":{"1":"done"}},
    {"name":"done","accept":true}
  ]
}`
	if err := os.WriteFile(tablePath, []byte(table), 0o644); err != nil {
		t.Fatal(err)
	}
	tokenizerPath := filepath.Join(dir, "tokenizer.json")
	tokenizer := `{
  "added_tokens":[{"id":0,"content":"<bos>","special":true},{"id":1,"content":"<eos>","special":true}],
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},
  "model":{"type":"BPE","vocab":{"<bos>":0,"<eos>":1,"a":2,"b":3,"c":4},"merges":[]}
}`
	if err := os.WriteFile(tokenizerPath, []byte(tokenizer), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name      string
		batchSize int
		configure func(*GenerateOptions)
	}{
		{name: "token_dfa_replay", batchSize: 1, configure: func(opts *GenerateOptions) { opts.GrammarTablePath = tablePath }},
		{name: "gbnf_batched", batchSize: 3, configure: func(opts *GenerateOptions) {
			opts.GrammarString = `root ::= "a"`
			opts.TokenizerPath = tokenizerPath
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			outputPath := filepath.Join(dir, test.name+".txt")
			eos := 1
			opts := GenerateOptions{
				ConfigPath: configPath, SafetensorsLoad: weightsPath,
				MaxTokens: 2, Temperature: 1, TopK: 1, Prompt: "token_ids:0",
				NumSamples: 3, GenerationBatch: test.batchSize, GenerationSeed: 7,
				EOSTokenID: &eos, OutputPath: outputPath,
			}
			test.configure(&opts)
			if err := runGenerateWithOptions(opts); err != nil {
				t.Fatal(err)
			}
			body, err := os.ReadFile(outputPath)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := strings.TrimSpace(string(body)), "0,2,1\n0,2,1\n0,2,1"; got != want {
				t.Fatalf("output=%q want=%q", got, want)
			}
		})
	}
}
