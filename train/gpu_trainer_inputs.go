//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func (t *mlxGPUTrainer) makeInputs(xTok, yTok []int, batchSize, seqLen int) ([]gpu.TensorInput, error) {
	return t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
}

func (t *mlxGPUTrainer) makeRTDGeneratorInputs(batch objectiveBatch) ([]gpu.TensorInput, error) {
	if !t.rtdGeneratorInput {
		return nil, fmt.Errorf("program does not declare rtd_generator inputs")
	}
	if t.rtdGeneratorBatchSize <= 0 || t.rtdGeneratorSeqLen <= 0 {
		return nil, fmt.Errorf("program declares rtd_generator_tokens with invalid shape [%d,%d]", t.rtdGeneratorBatchSize, t.rtdGeneratorSeqLen)
	}
	need := t.rtdGeneratorBatchSize * t.rtdGeneratorSeqLen
	selectedNeed := need
	if t.rtdGeneratorPositionsInput {
		if t.rtdGeneratorMaskSlots <= 0 {
			return nil, fmt.Errorf("program declares rtd_generator_positions with invalid shape [%d]", t.rtdGeneratorMaskSlots)
		}
		selectedNeed = t.rtdGeneratorBatchSize * t.rtdGeneratorMaskSlots
		if len(t.rtdGeneratorPosBuf) < t.rtdGeneratorMaskSlots {
			t.rtdGeneratorPosBuf = make([]int32, t.rtdGeneratorMaskSlots)
		}
	}
	if len(t.rtdGeneratorTokBuf) < need {
		t.rtdGeneratorTokBuf = make([]int32, need)
	}
	if len(t.rtdGeneratorTgtBuf) < selectedNeed {
		t.rtdGeneratorTgtBuf = make([]int32, selectedNeed)
		t.rtdGeneratorLossBuf = make([]float32, selectedNeed)
	}
	if len(batch.rtdGeneratorX) < need {
		return nil, fmt.Errorf("objective batch missing rtd_generator_tokens: got=%d need=%d", len(batch.rtdGeneratorX), need)
	}
	if t.rtdGeneratorPositionsInput && len(batch.rtdGeneratorPositions) < t.rtdGeneratorMaskSlots {
		return nil, fmt.Errorf("objective batch missing rtd_generator_positions: got=%d need=%d", len(batch.rtdGeneratorPositions), t.rtdGeneratorMaskSlots)
	}
	if len(batch.rtdGeneratorY) < selectedNeed {
		return nil, fmt.Errorf("objective batch missing rtd_generator_targets: got=%d need=%d", len(batch.rtdGeneratorY), selectedNeed)
	}
	if len(batch.rtdGeneratorLossMask) < selectedNeed {
		return nil, fmt.Errorf("objective batch missing rtd_generator_loss_mask: got=%d need=%d", len(batch.rtdGeneratorLossMask), selectedNeed)
	}
	for i := 0; i < need; i++ {
		t.rtdGeneratorTokBuf[i] = int32(batch.rtdGeneratorX[i])
	}
	for i := 0; i < selectedNeed; i++ {
		t.rtdGeneratorTgtBuf[i] = int32(batch.rtdGeneratorY[i])
	}
	copy(t.rtdGeneratorLossBuf[:selectedNeed], batch.rtdGeneratorLossMask[:selectedNeed])
	inputs := []gpu.TensorInput{
		{Name: "rtd_generator_tokens", DType: gpu.TensorInt32, Shape: []int{t.rtdGeneratorBatchSize, t.rtdGeneratorSeqLen}, Data: t.rtdGeneratorTokBuf[:need]},
	}
	if t.rtdGeneratorPositionsInput {
		copy(t.rtdGeneratorPosBuf[:t.rtdGeneratorMaskSlots], batch.rtdGeneratorPositions[:t.rtdGeneratorMaskSlots])
		inputs = append(inputs, gpu.TensorInput{Name: "rtd_generator_positions", DType: gpu.TensorInt32, Shape: []int{t.rtdGeneratorMaskSlots}, Data: t.rtdGeneratorPosBuf[:t.rtdGeneratorMaskSlots]})
	}
	inputs = append(inputs,
		gpu.TensorInput{Name: "rtd_generator_targets", DType: gpu.TensorInt32, Shape: []int{selectedNeed}, Data: t.rtdGeneratorTgtBuf[:selectedNeed]},
		gpu.TensorInput{Name: "rtd_generator_loss_mask", DType: gpu.TensorFloat32, Shape: []int{selectedNeed}, Data: t.rtdGeneratorLossBuf[:selectedNeed]},
	)
	return inputs, nil
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
		t.distillLossMaskBuf = make([]float32, need)
		t.wordStructTargetBuf = make([]int32, need)
		t.wordStructLossMaskBuf = make([]float32, need)
		t.invarianceLossMaskBuf = make([]float32, need)
		t.pllMarginLossMaskBuf = make([]float32, need)
		t.energySpanMaskBuf = make([]float32, need)
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
	rtdGeneratorNeed := 0
	rtdGeneratorSelectedNeed := 0
	rtdGeneratorBatchSize := t.rtdGeneratorBatchSize
	rtdGeneratorSeqLen := t.rtdGeneratorSeqLen
	if t.rtdGeneratorInput {
		if rtdGeneratorBatchSize <= 0 || rtdGeneratorSeqLen <= 0 {
			return nil, fmt.Errorf("program declares rtd_generator_tokens with invalid shape [%d,%d]", rtdGeneratorBatchSize, rtdGeneratorSeqLen)
		}
		rtdGeneratorNeed = rtdGeneratorBatchSize * rtdGeneratorSeqLen
		rtdGeneratorSelectedNeed = rtdGeneratorNeed
		if t.rtdGeneratorPositionsInput {
			if t.rtdGeneratorMaskSlots <= 0 {
				return nil, fmt.Errorf("program declares rtd_generator_positions with invalid shape [%d]", t.rtdGeneratorMaskSlots)
			}
			rtdGeneratorSelectedNeed = rtdGeneratorBatchSize * t.rtdGeneratorMaskSlots
			if len(t.rtdGeneratorPosBuf) < t.rtdGeneratorMaskSlots {
				t.rtdGeneratorPosBuf = make([]int32, t.rtdGeneratorMaskSlots)
			}
		}
		if len(t.rtdGeneratorTokBuf) < rtdGeneratorNeed {
			t.rtdGeneratorTokBuf = make([]int32, rtdGeneratorNeed)
		}
		if len(t.rtdGeneratorTgtBuf) < rtdGeneratorSelectedNeed {
			t.rtdGeneratorTgtBuf = make([]int32, rtdGeneratorSelectedNeed)
			t.rtdGeneratorLossBuf = make([]float32, rtdGeneratorSelectedNeed)
		}
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
	if t.dropoutKeyCount > 0 {
		needKeys := t.dropoutKeyCount * 2
		if len(t.dropoutKeyBuf) < needKeys {
			t.dropoutKeyBuf = make([]int32, needKeys)
		}
		fillDropoutKeys(t.dropoutKeyBuf[:needKeys], t.trainingSeed, t.trainingStep)
		inputs = append(inputs, gpu.TensorInput{
			Name: ir.DropoutKeysInput, DType: gpu.TensorInt32, Shape: []int{t.dropoutKeyCount, 2}, Data: t.dropoutKeyBuf[:needKeys],
		})
	}
	if t.tttInnerLRScaleInput {
		if t.tttInnerLRScaleCount <= 0 {
			return nil, fmt.Errorf("program declares ttt_inner_lr_scale with invalid length=%d", t.tttInnerLRScaleCount)
		}
		if len(t.tttInnerLRScaleBuf) < t.tttInnerLRScaleCount {
			t.tttInnerLRScaleBuf = make([]float32, t.tttInnerLRScaleCount)
		}
		for i := 0; i < t.tttInnerLRScaleCount; i++ {
			t.tttInnerLRScaleBuf[i] = 1
		}
		if len(batch.tttInnerLRScale) >= t.tttInnerLRScaleCount {
			copy(t.tttInnerLRScaleBuf[:t.tttInnerLRScaleCount], batch.tttInnerLRScale[:t.tttInnerLRScaleCount])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "ttt_inner_lr_scale", DType: gpu.TensorFloat32, Shape: []int{t.tttInnerLRScaleCount}, Data: t.tttInnerLRScaleBuf[:t.tttInnerLRScaleCount],
		})
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
	if t.distillLossMaskInput {
		if len(t.distillLossMaskBuf) < need {
			t.distillLossMaskBuf = make([]float32, need)
		}
		if len(batch.maskedLossMask) >= need {
			copy(t.distillLossMaskBuf[:need], batch.maskedLossMask[:need])
		} else {
			clear(t.distillLossMaskBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "distill_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.distillLossMaskBuf[:need],
		})
	}
	if t.wordStructInput {
		if len(t.wordStructTargetBuf) < need {
			t.wordStructTargetBuf = make([]int32, need)
			t.wordStructLossMaskBuf = make([]float32, need)
		}
		if len(batch.wordStructTargets) >= need {
			for i := 0; i < need; i++ {
				t.wordStructTargetBuf[i] = int32(batch.wordStructTargets[i])
			}
		} else {
			for i := 0; i < need; i++ {
				t.wordStructTargetBuf[i] = t.tgtBuf[i]
			}
		}
		if len(batch.wordStructLossMask) >= need {
			copy(t.wordStructLossMaskBuf[:need], batch.wordStructLossMask[:need])
		} else {
			clear(t.wordStructLossMaskBuf[:need])
		}
		inputs = append(inputs,
			gpu.TensorInput{Name: "word_struct_targets", DType: gpu.TensorInt32, Shape: []int{need}, Data: t.wordStructTargetBuf[:need]},
			gpu.TensorInput{Name: "word_struct_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.wordStructLossMaskBuf[:need]},
		)
	}
	if t.invarianceInput {
		if len(t.invarianceLossMaskBuf) < need {
			t.invarianceLossMaskBuf = make([]float32, need)
		}
		if len(batch.invarianceLossMask) >= need {
			copy(t.invarianceLossMaskBuf[:need], batch.invarianceLossMask[:need])
		} else {
			clear(t.invarianceLossMaskBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "invariance_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.invarianceLossMaskBuf[:need],
		})
	}
	if t.pllMarginInput {
		if len(t.pllMarginLossMaskBuf) < need {
			t.pllMarginLossMaskBuf = make([]float32, need)
		}
		if len(batch.pllMarginLossMask) >= need {
			copy(t.pllMarginLossMaskBuf[:need], batch.pllMarginLossMask[:need])
		} else {
			clear(t.pllMarginLossMaskBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "pll_margin_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.pllMarginLossMaskBuf[:need],
		})
	}
	if t.energySpanMaskInput {
		if len(t.energySpanMaskBuf) < need {
			t.energySpanMaskBuf = make([]float32, need)
		}
		if len(batch.energySpanMask) >= need {
			copy(t.energySpanMaskBuf[:need], batch.energySpanMask[:need])
		} else {
			clear(t.energySpanMaskBuf[:need])
		}
		inputs = append(inputs, gpu.TensorInput{
			Name: "energy_span_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: t.energySpanMaskBuf[:need],
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
	if t.rtdGeneratorInput {
		if len(batch.rtdGeneratorX) >= rtdGeneratorNeed {
			for i := 0; i < rtdGeneratorNeed; i++ {
				t.rtdGeneratorTokBuf[i] = int32(batch.rtdGeneratorX[i])
			}
		} else {
			clear(t.rtdGeneratorTokBuf[:rtdGeneratorNeed])
		}
		if t.rtdGeneratorPositionsInput {
			if len(batch.rtdGeneratorPositions) >= t.rtdGeneratorMaskSlots {
				copy(t.rtdGeneratorPosBuf[:t.rtdGeneratorMaskSlots], batch.rtdGeneratorPositions[:t.rtdGeneratorMaskSlots])
			} else {
				clear(t.rtdGeneratorPosBuf[:t.rtdGeneratorMaskSlots])
			}
		}
		if len(batch.rtdGeneratorY) >= rtdGeneratorSelectedNeed {
			for i := 0; i < rtdGeneratorSelectedNeed; i++ {
				t.rtdGeneratorTgtBuf[i] = int32(batch.rtdGeneratorY[i])
			}
		} else {
			clear(t.rtdGeneratorTgtBuf[:rtdGeneratorSelectedNeed])
		}
		if len(batch.rtdGeneratorLossMask) >= rtdGeneratorSelectedNeed {
			copy(t.rtdGeneratorLossBuf[:rtdGeneratorSelectedNeed], batch.rtdGeneratorLossMask[:rtdGeneratorSelectedNeed])
		} else {
			clear(t.rtdGeneratorLossBuf[:rtdGeneratorSelectedNeed])
		}
		inputs = append(inputs, gpu.TensorInput{Name: "rtd_generator_tokens", DType: gpu.TensorInt32, Shape: []int{rtdGeneratorBatchSize, rtdGeneratorSeqLen}, Data: t.rtdGeneratorTokBuf[:rtdGeneratorNeed]})
		if t.rtdGeneratorPositionsInput {
			inputs = append(inputs, gpu.TensorInput{Name: "rtd_generator_positions", DType: gpu.TensorInt32, Shape: []int{t.rtdGeneratorMaskSlots}, Data: t.rtdGeneratorPosBuf[:t.rtdGeneratorMaskSlots]})
		}
		inputs = append(inputs,
			gpu.TensorInput{Name: "rtd_generator_targets", DType: gpu.TensorInt32, Shape: []int{rtdGeneratorSelectedNeed}, Data: t.rtdGeneratorTgtBuf[:rtdGeneratorSelectedNeed]},
			gpu.TensorInput{Name: "rtd_generator_loss_mask", DType: gpu.TensorFloat32, Shape: []int{rtdGeneratorSelectedNeed}, Data: t.rtdGeneratorLossBuf[:rtdGeneratorSelectedNeed]},
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
