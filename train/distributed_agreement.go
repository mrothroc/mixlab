//go:build mlx && cgo && (darwin || linux)

package train

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
	"math"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func distributedInitializationFields(
	cfg *ArchConfig,
	program *arch.Program,
	shapes []WeightShape,
	optimizer gpu.TrainerOptimizerSpec,
	weightData [][]float32,
	context *DistributedTrainerContext,
	options gpu.DistributedTrainerOptions,
) ([]gpu.InitializationAgreementField, error) {
	if context == nil {
		return nil, fmt.Errorf("distributed initialization context is nil")
	}
	configHash, err := resumeConfigHash(cfg)
	if err != nil {
		return nil, fmt.Errorf("hash distributed config: %w", err)
	}
	fields := []gpu.InitializationAgreementField{
		{Name: "group_protocol", Value: hashString64("mixlab-ddp-r1-v1")},
		{Name: "membership", Value: hashString64(context.LocalView.Membership.MembersHash)},
		{Name: "generation", Value: context.LocalView.Membership.Generation},
		{Name: "backend", Value: hashString64(context.LocalView.Membership.Backend)},
		{Name: "world_size", Value: uint64(context.LocalView.Membership.WorldSize())},
		{Name: "config", Value: hashString64(configHash)},
	}
	for _, value := range []struct {
		name  string
		value any
	}{
		{name: "ir", value: program},
		{name: "weight_layout", value: shapes},
		{name: "optimizer", value: optimizer},
		{name: "bucket_metadata", value: struct {
			Bytes  uint64
			Shapes []WeightShape
		}{Bytes: options.GradientBucketBytes, Shapes: shapes}},
	} {
		digest, hashErr := hashJSON64(value.value)
		if hashErr != nil {
			return nil, fmt.Errorf("hash distributed %s: %w", value.name, hashErr)
		}
		fields = append(fields, gpu.InitializationAgreementField{
			Name:  value.name,
			Value: digest,
		})
	}
	fields = append(
		fields,
		gpu.InitializationAgreementField{
			Name:  "dataset",
			Value: hashString64(context.DatasetHash),
		},
		gpu.InitializationAgreementField{
			Name:  "initial_parameters",
			Value: hashWeightData64(weightData),
		},
		gpu.InitializationAgreementField{
			Name:  "accumulation_steps",
			Value: uint64(options.AccumulationSteps),
		},
		gpu.InitializationAgreementField{
			Name:  "phase",
			Value: hashString64(context.ScheduledPhase),
		},
	)
	return fields, nil
}

func hashJSON64(value any) (uint64, error) {
	blob, err := json.Marshal(value)
	if err != nil {
		return 0, err
	}
	return hashBytes64(blob), nil
}

func hashString64(value string) uint64 {
	return hashBytes64([]byte(value))
}

func hashBytes64(value []byte) uint64 {
	sum := sha256.Sum256(value)
	return binary.LittleEndian.Uint64(sum[:8])
}

func hashWeightData64(weights [][]float32) uint64 {
	digest := sha256.New()
	const chunkValues = 4096
	encoded := make([]byte, chunkValues*4)
	for _, weight := range weights {
		writeUint64(digest, uint64(len(weight)))
		for start := 0; start < len(weight); start += chunkValues {
			end := min(start+chunkValues, len(weight))
			for i, value := range weight[start:end] {
				binary.LittleEndian.PutUint32(
					encoded[i*4:(i+1)*4],
					math.Float32bits(value),
				)
			}
			_, _ = digest.Write(encoded[:(end-start)*4])
		}
	}
	return binary.LittleEndian.Uint64(digest.Sum(nil)[:8])
}

func writeUint64(dst hash.Hash, value uint64) {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	_, _ = dst.Write(encoded[:])
}
