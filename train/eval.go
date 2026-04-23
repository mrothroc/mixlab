package train

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/mrothroc/mixlab/data"
)

// bpbLUT holds the bytes-per-token lookup tables for BPB calculation.
type bpbLUT struct {
	baseBytes  []uint16
	hasLeading []bool
	isBoundary []bool
}

func loadUint16LUT(path string, expected int) ([]uint16, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(blob)%2 != 0 {
		return nil, fmt.Errorf("invalid uint16 LUT size for %s: %d", path, len(blob))
	}
	n := len(blob) / 2
	if n < expected {
		return nil, fmt.Errorf("LUT %s too short: got %d want >=%d", path, n, expected)
	}
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint16(blob[i*2:])
	}
	return out, nil
}

func loadBoolLUT(path string, expected int) ([]bool, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(blob) < expected {
		return nil, fmt.Errorf("LUT %s too short: got %d want >=%d", path, len(blob), expected)
	}
	out := make([]bool, len(blob))
	for i, v := range blob {
		out[i] = v != 0
	}
	return out, nil
}

func loadBPBLUTs(dir string, vocab int) (*bpbLUT, error) {
	base, err := loadUint16LUT(filepath.Join(dir, "bytes_per_token.bin"), vocab)
	if err != nil {
		return nil, err
	}
	hasLeading, err := loadBoolLUT(filepath.Join(dir, "has_leading_space.bin"), vocab)
	if err != nil {
		return nil, err
	}
	isBoundary, err := loadBoolLUT(filepath.Join(dir, "is_boundary_token.bin"), vocab)
	if err != nil {
		return nil, err
	}
	return &bpbLUT{
		baseBytes:  base,
		hasLeading: hasLeading,
		isBoundary: isBoundary,
	}, nil
}

// runFullEval performs full validation BPB evaluation on all validation shards.
func runFullEval(cfg *ArchConfig, valPattern string, trainer GPUTrainer, lutDir string) error {
	return runFullEvalWithTTT(
		cfg,
		valPattern,
		trainer,
		lutDir,
		cfg.Training.TTTMode,
		cfg.Training.TTTSteps,
		float32(cfg.Training.TTTLR),
		cfg.Training.TTTRank,
	)
}

// runFullEvalWithTTT performs score-first full validation BPB evaluation and,
// when enabled, adapts weights after each scored batch.
func runFullEvalWithTTT(
	cfg *ArchConfig,
	valPattern string,
	trainer GPUTrainer,
	lutDir string,
	tttMode string,
	tttSteps int,
	tttLR float32,
	tttRank int,
) error {
	name := cfg.Name
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	vocab := cfg.VocabSize
	if tttSteps < 0 {
		return fmt.Errorf("ttt_steps must be >= 0")
	}
	if tttMode == "" {
		tttMode = "full"
	}
	if tttMode != "full" && tttMode != "lora" {
		return fmt.Errorf("ttt_mode must be \"full\" or \"lora\"")
	}
	if tttMode == "lora" && tttRank <= 0 {
		return fmt.Errorf("ttt_rank must be > 0")
	}

	luts, err := loadBPBLUTs(lutDir, vocab)
	if err != nil {
		return fmt.Errorf("load BPB LUTs from %s: %w\n  generate LUTs with: mixlab -mode prepare, or set -lut-dir to the directory containing bytes_per_token.bin, has_leading_space.bin, is_boundary_token.bin", lutDir, err)
	}

	valTokens, err := data.LoadValidationTokens(valPattern, seqLen)
	if err != nil {
		return err
	}
	if len(valTokens) < 2 {
		return fmt.Errorf("validation tokens too short: %d", len(valTokens))
	}
	totalSeqs := (len(valTokens) - 1) / seqLen

	totalLossNats := 0.0
	totalBytes := 0.0
	totalTokens := 0
	rawStart := 0
	processedSeqs := 0
	evalStart := time.Now()
	lastProgress := evalStart

	for rawStart+1 < len(valTokens) {
		remaining := (len(valTokens) - 1) - rawStart
		chunkTokens := batchTokens
		if remaining < chunkTokens {
			chunkTokens = remaining - (remaining % seqLen)
		}
		if chunkTokens <= 0 {
			break
		}

		local := valTokens[rawStart : rawStart+chunkTokens+1]
		xTok := make([]int, chunkTokens)
		yTok := make([]int, chunkTokens)
		for i := 0; i < chunkTokens; i++ {
			xTok[i] = int(local[i])
			yTok[i] = int(local[i+1])
		}
		batchSize := chunkTokens / seqLen

		var lossV float32
		if tttMode == "lora" && tttSteps > 0 {
			lossV, err = trainer.EvaluateLoRATTTGPU(xTok, yTok, batchSize, seqLen, tttSteps, tttLR, tttRank)
		} else {
			lossV, err = trainer.EvaluateGPU(xTok, yTok, batchSize, seqLen)
		}
		if err != nil {
			return fmt.Errorf("eval failed at token offset %d: %w", rawStart, err)
		}
		meanLoss := float64(lossV)

		totalLossNats += meanLoss * float64(chunkTokens)
		for i := 0; i < chunkTokens; i++ {
			prevID := xTok[i]
			tgtID := yTok[i]
			if prevID < 0 || prevID >= len(luts.isBoundary) || tgtID < 0 || tgtID >= len(luts.baseBytes) || tgtID >= len(luts.hasLeading) {
				return fmt.Errorf("token id out of LUT bounds at offset %d: prev=%d tgt=%d", rawStart+i, prevID, tgtID)
			}
			tokenBytes := float64(luts.baseBytes[tgtID])
			if luts.hasLeading[tgtID] && !luts.isBoundary[prevID] {
				tokenBytes += 1
			}
			totalBytes += tokenBytes
		}
		totalTokens += chunkTokens
		if tttMode == "full" {
			for step := 0; step < tttSteps; step++ {
				if _, err := trainer.TrainStepGPU(xTok, yTok, batchSize, seqLen, tttLR); err != nil {
					return fmt.Errorf("ttt step %d failed at token offset %d: %w", step+1, rawStart, err)
				}
			}
		}
		rawStart += chunkTokens
		processedSeqs += chunkTokens / seqLen

		now := time.Now()
		if now.Sub(lastProgress) >= 30*time.Second || processedSeqs >= totalSeqs {
			pct := 0.0
			if totalSeqs > 0 {
				pct = 100.0 * float64(processedSeqs) / float64(totalSeqs)
			}
			elapsed := now.Sub(evalStart)
			etaStr := "n/a"
			if processedSeqs >= totalSeqs {
				etaStr = "0s"
			} else if processedSeqs > 0 {
				remainingSeqs := totalSeqs - processedSeqs
				eta := time.Duration((elapsed.Seconds() / float64(processedSeqs)) * float64(remainingSeqs) * float64(time.Second))
				etaStr = eta.Round(time.Second).String()
			}
			fmt.Printf("  [%s] full_val progress seq=%d/%d (%.1f%%) elapsed=%s eta=%s\n",
				name, processedSeqs, totalSeqs, pct, elapsed.Round(time.Second), etaStr)
			lastProgress = now
		}
	}
	if totalTokens == 0 || totalBytes <= 0 {
		return fmt.Errorf("invalid BPB totals: tokens=%d bytes=%f", totalTokens, totalBytes)
	}
	bpb := (totalLossNats / math.Log(2.0)) / totalBytes
	avgNLL := totalLossNats / float64(totalTokens)
	if tttSteps > 0 {
		if tttMode == "lora" {
			fmt.Printf("  [%s] full_val nll=%.6f bpb=%.6f tokens=%d bytes=%.0f ttt_mode=lora ttt_steps=%d ttt_lr=%g ttt_rank=%d\n", name, avgNLL, bpb, totalTokens, totalBytes, tttSteps, tttLR, tttRank)
		} else {
			fmt.Printf("  [%s] full_val nll=%.6f bpb=%.6f tokens=%d bytes=%.0f ttt_steps=%d ttt_lr=%g\n", name, avgNLL, bpb, totalTokens, totalBytes, tttSteps, tttLR)
		}
	} else {
		fmt.Printf("  [%s] full_val nll=%.6f bpb=%.6f tokens=%d bytes=%.0f\n", name, avgNLL, bpb, totalTokens, totalBytes)
	}
	return nil
}
