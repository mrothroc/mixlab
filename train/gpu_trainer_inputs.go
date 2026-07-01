//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

func (t *mlxGPUTrainer) makeInputs(xTok, yTok []int, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	return t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
}

func (t *mlxGPUTrainer) makeObjectiveInputs(batch objectiveBatch, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	need := batchSize * seqLen
	if len(batch.x) < need || len(batch.y) < need {
		return nil, fmt.Errorf("input size mismatch: tokens=%d targets=%d need=%d", len(batch.x), len(batch.y), need)
	}
	// Grow buffers if needed (shouldn't happen with consistent batch size).
	if len(t.tokBuf) < need {
		t.tokBuf = make([]int32, need)
		t.tgtBuf = make([]int32, need)
		t.lossMaskBuf = make([]float32, need)
		t.attentionCausalBuf = make([]int32, need)
		t.segmentIDBuf = make([]int32, need)
		t.charBuf = make([]int32, need*t.charMaxPerToken)
		t.bigramBuf = make([]int32, need)
		t.trigramBuf = make([]int32, need)
	}
	if t.attentionCausalInput && len(t.attentionCausalBuf) < batchSize {
		t.attentionCausalBuf = make([]int32, batchSize)
	}
	if t.segmentIDsInput && len(t.segmentIDBuf) < need {
		t.segmentIDBuf = make([]int32, need)
	}
	if t.diffusionBlockStartInput && len(t.diffusionBlockStartBuf) < batchSize {
		t.diffusionBlockStartBuf = make([]int32, batchSize)
	}
	if t.diffusionBlockEndInput && len(t.diffusionBlockEndBuf) < batchSize {
		t.diffusionBlockEndBuf = make([]int32, batchSize)
	}
	if t.diffusionTimestepInput && len(t.diffusionTimestepBuf) < batchSize {
		t.diffusionTimestepBuf = make([]float32, batchSize)
	}
	if t.charInput && len(t.charBuf) < need*t.charMaxPerToken {
		t.charBuf = make([]int32, need*t.charMaxPerToken)
	}
	needTeacherProbs := need * t.vocabSize
	if t.teacherProbsInput && len(t.teacherProbBuf) < needTeacherProbs {
		t.teacherProbBuf = make([]float32, needTeacherProbs)
	}
	needData2VecTargets := need * t.shapesModelDim()
	if t.data2VecInput && len(t.data2VecTargetBuf) < needData2VecTargets {
		t.data2VecTargetBuf = make([]float32, needData2VecTargets)
	}
	if t.data2VecInput && len(t.data2VecMaskBuf) < need {
		t.data2VecMaskBuf = make([]float32, need)
	}
	for i := 0; i < need; i++ {
		t.tokBuf[i] = int32(batch.x[i])
		t.tgtBuf[i] = int32(batch.y[i])
	}
	targetData, targetShape, err := t.prepareTargets(batchSize, seqLen, need)
	if err != nil {
		return nil, err
	}
	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.tokBuf[:need]},
		{Name: "targets", DType: gpu.TensorInt32, Shape: targetShape, Data: targetData},
	}
	if t.lossMaskInput {
		if len(batch.lossMask) >= need {
			copy(t.lossMaskBuf[:need], batch.lossMask[:need])
		} else {
			for i := 0; i < need; i++ {
				t.lossMaskBuf[i] = 1
			}
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.lossMaskBuf[:need],
		})
	}
	if t.attentionCausalInput {
		if len(batch.attentionCausal) >= batchSize {
			copy(t.attentionCausalBuf[:batchSize], batch.attentionCausal[:batchSize])
		} else {
			for i := 0; i < batchSize; i++ {
				t.attentionCausalBuf[i] = 1
			}
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "attention_causal_mask", DType: gpu.TensorInt32, Shape: []int{batchSize}, Data: t.attentionCausalBuf[:batchSize],
		})
	}
	if t.segmentIDsInput {
		if len(batch.segmentIDs) >= need {
			copy(t.segmentIDBuf[:need], batch.segmentIDs[:need])
		} else {
			clear(t.segmentIDBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "segment_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.segmentIDBuf[:need],
		})
	}
	if t.diffusionBlockStartInput {
		if len(batch.diffusionBlockStart) < batchSize {
			return nil, fmt.Errorf("objective batch missing diffusion_block_start: got=%d need=%d", len(batch.diffusionBlockStart), batchSize)
		}
		copy(t.diffusionBlockStartBuf[:batchSize], batch.diffusionBlockStart[:batchSize])
		inputs = append(inputs, gpu.TensorInput{
			Name: "diffusion_block_start", DType: gpu.TensorInt32, Shape: []int{batchSize}, Data: t.diffusionBlockStartBuf[:batchSize],
		})
	}
	if t.diffusionBlockEndInput {
		if len(batch.diffusionBlockEnd) < batchSize {
			return nil, fmt.Errorf("objective batch missing diffusion_block_end: got=%d need=%d", len(batch.diffusionBlockEnd), batchSize)
		}
		copy(t.diffusionBlockEndBuf[:batchSize], batch.diffusionBlockEnd[:batchSize])
		inputs = append(inputs, gpu.TensorInput{
			Name: "diffusion_block_end", DType: gpu.TensorInt32, Shape: []int{batchSize}, Data: t.diffusionBlockEndBuf[:batchSize],
		})
	}
	if t.diffusionTimestepInput {
		if len(batch.diffusionTimestep) >= batchSize {
			copy(t.diffusionTimestepBuf[:batchSize], batch.diffusionTimestep[:batchSize])
		} else {
			clear(t.diffusionTimestepBuf[:batchSize])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "diffusion_timestep", DType: gpu.TensorFloat32, Shape: []int{batchSize}, Data: t.diffusionTimestepBuf[:batchSize],
		})
	}
	if t.teacherProbsInput {
		if t.vocabSize <= 0 {
			return nil, fmt.Errorf("invalid vocab size=%d", t.vocabSize)
		}
		if len(batch.teacherProbs) >= needTeacherProbs {
			copy(t.teacherProbBuf[:needTeacherProbs], batch.teacherProbs[:needTeacherProbs])
		} else {
			fillUniformTeacherProbs(t.teacherProbBuf[:needTeacherProbs], t.vocabSize)
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "teacher_probs", DType: gpu.TensorFloat32, Shape: []int{need, t.vocabSize}, Data: t.teacherProbBuf[:needTeacherProbs],
		})
	}
	if t.data2VecInput {
		modelDim := t.shapesModelDim()
		if modelDim <= 0 {
			return nil, fmt.Errorf("invalid model_dim for data2vec inputs")
		}
		if len(batch.data2vecTargets) >= needData2VecTargets {
			copy(t.data2VecTargetBuf[:needData2VecTargets], batch.data2vecTargets[:needData2VecTargets])
		} else {
			for i := 0; i < needData2VecTargets; i++ {
				t.data2VecTargetBuf[i] = 0
			}
		}
		if len(batch.data2vecMask) >= need {
			copy(t.data2VecMaskBuf[:need], batch.data2vecMask[:need])
		} else {
			for i := 0; i < need; i++ {
				t.data2VecMaskBuf[i] = 0
			}
		}
		inputs = append(inputs,
			gpu.TensorInput{Name: "data2vec_targets", DType: gpu.TensorFloat32, Shape: []int{need, modelDim}, Data: t.data2VecTargetBuf[:needData2VecTargets]},
			gpu.TensorInput{Name: "data2vec_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.data2VecMaskBuf[:need]},
		)
	}
	if t.firstByteMaskInput {
		if len(t.firstByteValid) != t.vocabSize {
			return nil, fmt.Errorf("first-byte mask size=%d does not match vocab_size=%d", len(t.firstByteValid), t.vocabSize)
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "first_byte_valid", DType: gpu.TensorInt32, Shape: []int{t.vocabSize}, Data: t.firstByteValid,
		})
	}
	if t.charInput {
		if t.charMaxPerToken <= 0 {
			return nil, fmt.Errorf("invalid char_max_per_token=%d", t.charMaxPerToken)
		}
		want := t.vocabSize * t.charMaxPerToken
		if len(t.charFeatures) != want {
			return nil, fmt.Errorf("char feature lookup size=%d does not match vocab_size*char_max_per_token=%d", len(t.charFeatures), want)
		}
		for i := 0; i < need; i++ {
			tok := int(t.tokBuf[i])
			if tok < 0 || tok >= t.vocabSize {
				return nil, fmt.Errorf("token id %d at position %d outside vocab_size=%d", tok, i, t.vocabSize)
			}
			copy(t.charBuf[i*t.charMaxPerToken:(i+1)*t.charMaxPerToken], t.charFeatures[tok*t.charMaxPerToken:(tok+1)*t.charMaxPerToken])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "char_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen, t.charMaxPerToken}, Data: t.charBuf[:need*t.charMaxPerToken],
		})
	}
	if t.bigramVocabSize > 0 {
		bigramIDs, err := ComputeBigramIDs(t.tokBuf[:need], need, t.bigramVocabSize)
		if err != nil {
			return nil, err
		}
		copy(t.bigramBuf[:need], bigramIDs)
		inputs = append(inputs, gpu.TensorInput{
			Name: "bigram_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.bigramBuf[:need],
		})
	}
	if t.trigramVocabSize > 0 {
		trigramIDs, err := ComputeTrigramIDs(t.tokBuf[:need], batchSize, seqLen, t.trigramVocabSize)
		if err != nil {
			return nil, err
		}
		copy(t.trigramBuf[:need], trigramIDs)
		inputs = append(inputs, gpu.TensorInput{
			Name: "trigram_ids", DType: gpu.TensorInt32, Shape: []int{batchSize, seqLen}, Data: t.trigramBuf[:need],
		})
	}
	return inputs, nil
}

func (t *mlxGPUTrainer) prepareTargets(batchSize, seqLen, targetSize int) ([]int32, []int, error) {
	if t.declaredTargetSize <= 0 || t.declaredTargetSize == targetSize {
		return t.tgtBuf[:targetSize], []int{targetSize}, nil
	}
	if t.declaredTargetSize < targetSize {
		return nil, nil, fmt.Errorf("declared targets size=%d smaller than batch target size=%d", t.declaredTargetSize, targetSize)
	}
	if batchSize <= 0 {
		return nil, nil, fmt.Errorf("invalid batch size=%d", batchSize)
	}
	extraTargets := t.declaredTargetSize - targetSize
	if extraTargets%batchSize != 0 {
		return nil, nil, fmt.Errorf("declared targets size=%d adds %d extras not divisible by batch size=%d", t.declaredTargetSize, extraTargets, batchSize)
	}
	kPerBatch := extraTargets / batchSize
	if kPerBatch <= 0 {
		return t.tgtBuf[:targetSize], []int{targetSize}, nil
	}
	if seqLen%kPerBatch != 0 {
		return nil, nil, fmt.Errorf("invalid extended targets: seq len=%d not divisible by extra targets per batch=%d", seqLen, kPerBatch)
	}
	stride := seqLen / kPerBatch
	extBuf := make([]int32, t.declaredTargetSize)
	copy(extBuf, t.tgtBuf[:targetSize])
	dst := targetSize
	for b := 0; b < batchSize; b++ {
		base := b * seqLen
		for i := 0; i < kPerBatch; i++ {
			extBuf[dst] = t.tgtBuf[base+i*stride]
			dst++
		}
	}
	return extBuf, []int{t.declaredTargetSize}, nil
}
