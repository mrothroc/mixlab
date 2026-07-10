package arch

import "fmt"

func buildMultiheadTrainingIRProgramFromConfig(cfg *ArchConfig, state TrainingProgramState) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	rawBatch := 1
	if cfg.Training.BatchTokens > 0 && cfg.SeqLen > 0 {
		rawBatch = cfg.Training.BatchTokens / cfg.SeqLen
		if rawBatch <= 0 {
			rawBatch = 1
		}
	}
	headCount := len(cfg.Training.Heads)
	if headCount <= 0 {
		return nil, fmt.Errorf("multihead objective has no heads")
	}
	normalBatchRows := rawBatch * headCount
	extraPairRows := 0
	if multiheadUsesMLMSpanPLL(cfg) {
		extraPairRows = rawBatch
	}
	B := normalBatchRows + extraPairRows
	T := cfg.SeqLen
	D := cfg.ModelDim
	V := cfg.VocabSize
	rawRows := rawBatch * T
	pairRowsStart := normalBatchRows * T
	nWeightsMeta, err := collectMultiheadWeightShapesFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	nWeights := len(nWeightsMeta)
	invarianceEnabled := cfg.Training.InvarianceActive() && !state.InvarianceInactive
	prog := NewProgram(nWeights)
	prog.DeclareInput("tokens", TensorInt32, []int{B, T})
	prog.DeclareInput("targets", TensorInt32, []int{B * T})
	prog.DeclareInput("loss_mask", TensorFloat32, []int{B * T})
	if multiheadUsesWordStructural(cfg) {
		prog.DeclareInput("word_struct_targets", TensorInt32, []int{B * T})
		prog.DeclareInput("word_struct_loss_mask", TensorFloat32, []int{B * T})
	}
	if invarianceEnabled {
		prog.DeclareInput("invariance_loss_mask", TensorFloat32, []int{B * T})
	}
	prog.DeclareInput("diffusion_block_start", TensorInt32, []int{B})
	prog.DeclareInput("diffusion_block_end", TensorInt32, []int{B})
	if multiheadUsesMinimalPairSpanMask(cfg) {
		prog.DeclareInput("energy_span_mask", TensorFloat32, []int{B * T})
	}
	if multiheadUsesAdaLN(cfg.Training) {
		prog.DeclareInput("diffusion_timestep", TensorFloat32, []int{B})
	}

	wi := 0
	prog.Embed(weightName(wi), "tokens", "x_embed")
	wi++
	embedState := "x_embed"
	positionalEmbedding := cfg.EffectivePositionalEmbedding()
	maxPositions := cfg.EffectiveMaxPositions()
	var emitErr error
	if positionalEmbedding == PositionalEmbeddingLearnedAbsolute {
		embedState, wi, emitErr = emitLearnedPositionEmbeddingIR(prog, embedState, B, T, D, wi, maxPositions)
		if emitErr != nil {
			return nil, emitErr
		}
	}
	xState := "x"
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 {
		xState = "x_tok"
	}
	prog.Reshape(embedState, []int{B * T, D}, xState)
	featureBase := xState
	wi = emitCharIR(prog, featureBase, B, T, D, wi, cfg.CharVocabSize, cfg.EffectiveCharDim(), cfg.EffectiveCharMaxPerToken())
	if cfg.CharVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitBigramIR(prog, featureBase, B, T, D, wi, cfg.BigramVocabSize, cfg.EffectiveBigramDim())
	if cfg.BigramVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitTrigramIR(prog, featureBase, B, T, D, wi, cfg.TrigramVocabSize, cfg.EffectiveTrigramDim())
	embeddingDropout := cfg.EffectiveEmbeddingDropout()
	hiddenDropout := cfg.EffectiveHiddenDropout()
	attnDropout := cfg.EffectiveAttnDropout()
	if state.DropoutInactive {
		embeddingDropout = 0
		hiddenDropout = 0
		attnDropout = 0
	}
	if embeddingDropout > 0 {
		prog.Dropout("x", embeddingDropout, "x")
	}
	sharedRel, err := newSharedRelativeAttentionPlan(cfg.Blocks)
	if err != nil {
		return nil, err
	}
	if sharedRel.Enabled {
		sharedRel.WeightIndex = wi
		wi++
		sharedRel.NormEps = cfg.EffectiveNormSpec().Eps
		if sharedRel.Norm == RelativeAttentionEmbeddingNormLayerNorm {
			sharedRel.NormIndex = wi
			wi += 2
		}
	}
	if cfg.ResidMix {
		prog.ScalarMul("x", 1.0, "x0")
	}

	blocks := multiheadResolvedBlocks(cfg.Blocks)
	blockWeightStart := wi
	blockShapes, err := multiheadTrunkWeightShapes(cfg)
	if err != nil {
		return nil, err
	}
	adalnStart := blockWeightStart + len(blockShapes)
	adalnWeights := 0
	adaLN := (*adaLNBuildState)(nil)
	if multiheadUsesAdaLN(cfg.Training) {
		adaLN = newAdaLNBuildState(true, adalnStart, multiheadAdaLNDim(cfg.Training), D)
		adalnWeights = len(blocks) * adaLN.weightsPerBlock()
	}
	captureDWA := multiheadAnyDWAHead(cfg.Training)
	var captureAgg *layerAggregationBuildState
	if captureDWA {
		captureAgg = newLayerAggregationCaptureState(prog, "x")
	}
	kvCache := make(map[int]BlockKVOutputs, len(blocks))
	opIdx := 0
	parallelPlan, err := newParallelResidualPlan(blocks, false)
	if err != nil {
		return nil, err
	}
	if parallelPlan.any {
		if adaLN != nil {
			return nil, fmt.Errorf("multihead parallel_group does not support diffusion AdaLN in v1")
		}
		refs := identityWeightRefs(blocks)
		weightStarts := make([]int, len(blocks))
		for i := range weightStarts {
			weightStarts[i] = -1
		}
		wi, err = emitSequentialRangeWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, 0, len(blocks), "x", "x0", wi, D, T, B, V, &opIdx, nil, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix, false, hiddenDropout, attnDropout, nil, cfg.EffectiveNormSpec(), cfg.EffectiveNormPlacement(), cfg.FFNInternalNorm, positionalEmbedding, sharedRel, captureAgg, false)
		if err != nil {
			return nil, err
		}
	} else {
		for i, spec := range blocks {
			if needsResidMix(spec, cfg.ResidMix) {
				wi = applyResidMixIR(prog, "x", "x0", wi, D, opIdx)
			}
			wi, err = EmitBlock(prog, spec, "x", wi, D, T, B, V, opIdx, EmitOptions{
				MLPMult:             cfg.EffectiveMLPMult(),
				BlockScales:         cfg.BlockScales,
				Dropout:             hiddenDropout,
				AttnDropout:         attnDropout,
				Norm:                cfg.EffectiveNormSpec(),
				NormPlacement:       cfg.EffectiveNormPlacement(),
				FFNInternalNorm:     cfg.FFNInternalNorm,
				PositionalEmbedding: positionalEmbedding,
				BlockIndex:          i,
				KVCache:             kvCache,
				sharedRelative:      sharedRel,
				layerAgg:            captureAgg,
				adaLN:               adaLN,
			})
			if err != nil {
				return nil, err
			}
			opIdx++
		}
	}
	if wi != blockWeightStart+len(blockShapes) {
		return nil, fmt.Errorf("multihead trunk weight count mismatch: emitted=%d expected=%d", wi-blockWeightStart, len(blockShapes))
	}
	wi += adalnWeights

	exportHeadName := cfg.Training.ExportHead
	diffusionHeadName := cfg.Training.DiffusionHead
	minimalPairPLL := multiheadUsesMLMSpanPLL(cfg)
	minimalPairScoreHead := ""
	if minimalPairPLL {
		minimalPairScoreHead = cfg.Training.MinimalPair.ScoreHead
	}
	wordStructuralEnabled := multiheadUsesWordStructural(cfg)
	invarianceHeadName := ""
	if invarianceEnabled {
		if head := cfg.Training.MultiheadExportHead(); head != nil {
			invarianceHeadName = head.Name
		}
	}
	lossAccum := ""
	norm := cfg.EffectiveNormSpec()
	for headIdx, head := range cfg.Training.Heads {
		rowStart := headIdx * rawRows
		headPrefix := "head_" + head.Name
		headIn := headPrefix + "_hidden"
		dwaWeightIndex := wi
		if head.LayerAggregation == LayerAggregationDWA {
			if captureAgg == nil {
				return nil, fmt.Errorf("head %q requested DWA without captured trunk states", head.Name)
			}
			var err error
			headIn, err = emitLayerAggregationFromHistory(prog, captureAgg.history, wi, rowStart, rawRows, headPrefix+"_dwa")
			if err != nil {
				return nil, err
			}
			wi++
		} else {
			prog.Slice("x", rowStart, rowStart+rawRows, 1, 0, headIn)
		}
		headMask := headPrefix + "_loss_mask"
		prog.Slice("loss_mask", rowStart, rowStart+rawRows, 1, 0, headMask)
		energySpanMode := head.Objective == ObjectiveEnergy && cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.UsesDifferingSpanEnergy()
		headParamStart := wi
		if head.Objective == ObjectiveEnergy {
			if cfg.Training.MinimalPair == nil {
				return nil, fmt.Errorf("energy head %q requires training.minimal_pair", head.Name)
			}
			if !energySpanMode {
				seqHidden := headPrefix + "_sequence_hidden"
				prog.Reshape(headIn, []int{rawBatch, T, D}, headPrefix+"_hidden_btd")
				prog.MeanAxis(headPrefix+"_hidden_btd", 1, seqHidden)
				headIn = seqHidden
			}
		}
		if head.FinalNorm {
			normOut := headPrefix + "_final_norm"
			wi, err = emitNamedNormIR(prog, headIn, wi, normOut, norm)
			if err != nil {
				return nil, err
			}
			headIn = normOut
		}
		logits, nextWI, err := emitMultiheadHeadLogitsIR(prog, headPrefix, head, headIn, wi, norm.Eps, hiddenDropout)
		if err != nil {
			return nil, err
		}
		wi = nextWI
		if cfg.LogitSoftcap > 0 {
			scaled := headPrefix + "_logits_softcap_scaled"
			tanh := headPrefix + "_logits_softcap_tanh"
			capped := headPrefix + "_logits_softcapped"
			prog.ScalarMul(logits, 1.0/cfg.LogitSoftcap, scaled)
			prog.Tanh(scaled, tanh)
			prog.ScalarMul(tanh, cfg.LogitSoftcap, capped)
			logits = capped
		}
		outputLogits := logits
		targets := headPrefix + "_targets"
		mask := headMask
		loss := headPrefix + "_loss"
		weighted := headPrefix + "_loss_weighted"
		if head.Objective != ObjectiveEnergy {
			prog.Slice("targets", rowStart, rowStart+rawRows, 1, 0, targets)
		}
		lossWeight := float32(head.LossWeight)
		switch head.Objective {
		case ObjectiveEnergy:
			acc := headPrefix + "_pair_accuracy"
			cleanMean := headPrefix + "_clean_energy_mean"
			corruptMean := headPrefix + "_corrupt_energy_mean"
			if energySpanMode {
				spanMask := headPrefix + "_span_mask"
				prog.Slice("energy_span_mask", rowStart, rowStart+rawRows, 1, 0, spanMask)
				prog.EnergySpanPairwiseLoss(logits, spanMask, T, cfg.Training.MinimalPair.LossKind(), float32(cfg.Training.MinimalPair.Margin), loss, acc, cleanMean, corruptMean)
				pooled := headPrefix + "_span_pooled_logits"
				prog.EnergySpanPool(logits, spanMask, T, pooled)
				outputLogits = pooled
				tokenEnergy := headPrefix + "_token_energy"
				prog.ScalarMul(logits, 1.0, tokenEnergy)
				prog.DeclareOutput(tokenEnergy, TensorFloat32, []int{rawRows, 1})
			} else {
				rowMask := headPrefix + "_row_mask"
				prog.Slice(mask, 0, rawRows, T, 0, rowMask)
				prog.EnergyPairwiseLoss(logits, rowMask, cfg.Training.MinimalPair.LossKind(), float32(cfg.Training.MinimalPair.Margin), loss, acc, cleanMean, corruptMean)
			}
			prog.DeclareOutput(acc, TensorFloat32, []int{1})
			prog.DeclareOutput(cleanMean, TensorFloat32, []int{1})
			prog.DeclareOutput(corruptMean, TensorFloat32, []int{1})
		case ObjectiveRTD:
			prog.MaskedBCEWithLogits(logits, targets, mask, loss)
			acc := headPrefix + "_accuracy"
			prog.MaskedBinaryAccuracy(logits, targets, mask, acc)
			prog.DeclareOutput(acc, TensorFloat32, []int{1})
			if cfg.Training.RTD != nil {
				lossWeight *= float32(cfg.Training.RTD.DiscriminatorLossWeight)
			}
		default:
			prog.MaskedCrossEntropy(logits, targets, mask, loss)
		}
		prog.ScalarMul(loss, lossWeight, weighted)
		if lossAccum == "" {
			lossAccum = weighted
		} else {
			next := headPrefix + "_loss_accum"
			prog.Add(lossAccum, weighted, next)
			lossAccum = next
		}
		prog.DeclareOutput(loss, TensorFloat32, []int{1})
		if head.Name == exportHeadName {
			prog.MaskedCrossEntropyPerToken(logits, targets, mask, "per_token_nll")
			prog.ScalarMul(loss, 1.0, "eval_loss")
			prog.ScalarMul(logits, 1.0, "logits")
		}
		generatorOutputName := ""
		if cfg.Training.RTD != nil {
			generatorOutputName = cfg.Training.RTD.GeneratorHead
		}
		if head.Name == exportHeadName || head.Name == diffusionHeadName || head.Name == generatorOutputName || (minimalPairPLL && head.Name == minimalPairScoreHead) {
			prog.ScalarMul(outputLogits, 1.0, headPrefix+"_logits")
			outCols := V
			outRows := rawRows
			switch head.Objective {
			case ObjectiveRTD:
				outCols = 1
			case ObjectiveEnergy:
				outCols = 1
				outRows = rawBatch
			}
			prog.DeclareOutput(headPrefix+"_logits", TensorFloat32, []int{outRows, outCols})
		} else {
			switch head.Objective {
			case ObjectiveRTD:
				prog.ScalarMul(outputLogits, 1.0, headPrefix+"_logits")
				prog.DeclareOutput(headPrefix+"_logits", TensorFloat32, []int{rawRows, 1})
			case ObjectiveEnergy:
				prog.ScalarMul(outputLogits, 1.0, headPrefix+"_logits")
				prog.DeclareOutput(headPrefix+"_logits", TensorFloat32, []int{rawBatch, 1})
			}
		}
		if minimalPairPLL && head.Name == minimalPairScoreHead {
			pairLoss, err := emitMultiheadMinimalPairPLLIR(prog, cfg, head, captureAgg, dwaWeightIndex, headParamStart, pairRowsStart, rawRows, T, norm, hiddenDropout)
			if err != nil {
				return nil, err
			}
			weightedPLL := headPrefix + "_minimal_pair_loss_weighted"
			prog.ScalarMul(pairLoss, float32(cfg.Training.MinimalPair.LossWeight), weightedPLL)
			next := headPrefix + "_minimal_pair_loss_accum"
			prog.Add(lossAccum, weightedPLL, next)
			lossAccum = next
		}
		if wordStructuralEnabled && cfg.Training.WordStructuralHeadSelected(head.Name) {
			wordTargets := headPrefix + "_word_struct_targets"
			wordMask := headPrefix + "_word_struct_loss_mask"
			wordLoss := headPrefix + "_word_struct_loss"
			prog.Slice("word_struct_targets", rowStart, rowStart+rawRows, 1, 0, wordTargets)
			prog.Slice("word_struct_loss_mask", rowStart, rowStart+rawRows, 1, 0, wordMask)
			prog.MaskedCrossEntropy(logits, wordTargets, wordMask, wordLoss)
			prog.ScalarMul(wordLoss, float32(cfg.Training.WordStructuralObjective.LossWeight), wordLoss+"_weighted")
			next := headPrefix + "_word_struct_loss_accum"
			prog.Add(lossAccum, wordLoss+"_weighted", next)
			lossAccum = next
			prog.DeclareOutput(wordLoss, TensorFloat32, []int{1})
		}
		if invarianceEnabled && head.Name == invarianceHeadName {
			invarianceMask := headPrefix + "_invariance_loss_mask"
			invarianceLoss := headPrefix + "_invariance_loss"
			prog.Slice("invariance_loss_mask", rowStart, rowStart+rawRows, 1, 0, invarianceMask)
			prog.MaskedSymmetricKL(logits, invarianceMask, T, invarianceLoss)
			prog.ScalarMul(invarianceLoss, float32(cfg.Training.Invariance.Weight), invarianceLoss+"_weighted")
			next := headPrefix + "_invariance_loss_accum"
			prog.Add(lossAccum, invarianceLoss+"_weighted", next)
			lossAccum = next
			prog.DeclareOutput(invarianceLoss, TensorFloat32, []int{1})
		}
	}
	prog.Slice("x", 0, rawRows, 1, 0, "x_hidden_flat")
	prog.Reshape("x_hidden_flat", []int{rawBatch, T, D}, "x_hidden")
	if cfg.Training.RTDDedicatedGeneratorEnabled() {
		var generatorLogits string
		generatorLogits, _, wi, err = emitRTDDedicatedGeneratorIR(prog, cfg, wi, rawBatch, hiddenDropout)
		if err != nil {
			return nil, err
		}
		prog.MaskedCrossEntropy(generatorLogits, "rtd_generator_targets", "rtd_generator_loss_mask", "rtd_generator_loss")
		prog.ScalarMul("rtd_generator_loss", float32(cfg.Training.RTD.DedicatedGenerator.GeneratorLossWeight), "rtd_generator_loss_weighted")
		if lossAccum == "" {
			lossAccum = "rtd_generator_loss_weighted"
		} else {
			prog.Add(lossAccum, "rtd_generator_loss_weighted", "rtd_generator_loss_accum")
			lossAccum = "rtd_generator_loss_accum"
		}
		prog.DeclareOutput("rtd_generator_loss", TensorFloat32, []int{1})
		generatorRows := rawBatch * RTDDedicatedGeneratorMaskSlots(cfg, T)
		prog.DeclareOutput(generatorLogits, TensorFloat32, []int{generatorRows, V})
	}
	if lossAccum == "" {
		return nil, fmt.Errorf("multihead graph emitted no losses")
	}
	prog.ScalarMul(lossAccum, 1.0, "loss")
	moeEnabled := emitMoEAuxiliaryAggregatesIR(prog, true)
	prog.DeclareOutput("loss", TensorFloat32, []int{1})
	prog.DeclareOutput("eval_loss", TensorFloat32, []int{1})
	prog.DeclareOutput("per_token_nll", TensorFloat32, []int{rawRows})
	prog.DeclareOutput("logits", TensorFloat32, []int{rawRows, V})
	if moeEnabled {
		prog.DeclareOutput("moe_aux_loss", TensorFloat32, []int{1})
		prog.DeclareOutput("moe_router_entropy", TensorFloat32, []int{1})
	}
	prog.DeclareOutput("x_hidden", TensorFloat32, []int{rawBatch, T, D})
	if wi != nWeights {
		return nil, fmt.Errorf("multihead IR weight count mismatch: emitted=%d expected=%d", wi, nWeights)
	}
	return prog, nil
}

func multiheadUsesMinimalPairSpanMask(cfg *ArchConfig) bool {
	return multiheadUsesDifferingSpanEnergy(cfg) || multiheadUsesMLMSpanPLL(cfg)
}

func multiheadUsesDifferingSpanEnergy(cfg *ArchConfig) bool {
	if cfg == nil || cfg.Training.MinimalPair == nil || !cfg.Training.MinimalPair.UsesDifferingSpanEnergy() {
		return false
	}
	for _, head := range cfg.Training.Heads {
		if head.Objective == ObjectiveEnergy {
			return true
		}
	}
	return false
}

func multiheadUsesMLMSpanPLL(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.UsesMLMSpanPLL()
}

func multiheadUsesWordStructural(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MultiheadEnabled() && cfg.Training.WordStructuralActive() && len(cfg.Training.WordStructuralSelectedHeads()) > 0
}

func emitMultiheadMinimalPairPLLIR(prog *Program, cfg *ArchConfig, head MultiheadHeadSpec, captureAgg *layerAggregationBuildState, dwaWeightIndex, headParamStart, rowStart, rows, seqLen int, norm NormSpec, dropout float32) (string, error) {
	prefix := "head_" + head.Name + "_minimal_pair"
	headIn := prefix + "_hidden"
	if head.LayerAggregation == LayerAggregationDWA {
		if captureAgg == nil {
			return "", fmt.Errorf("head %q requested DWA without captured trunk states", head.Name)
		}
		var err error
		headIn, err = emitLayerAggregationFromHistory(prog, captureAgg.history, dwaWeightIndex, rowStart, rows, prefix+"_dwa")
		if err != nil {
			return "", err
		}
	} else {
		prog.Slice("x", rowStart, rowStart+rows, 1, 0, headIn)
	}
	pairWI := headParamStart
	var err error
	if head.FinalNorm {
		normOut := prefix + "_final_norm"
		pairWI, err = emitNamedNormIR(prog, headIn, pairWI, normOut, norm)
		if err != nil {
			return "", err
		}
		headIn = normOut
	}
	logits, _, err := emitMultiheadHeadLogitsIR(prog, prefix, head, headIn, pairWI, norm.Eps, dropout)
	if err != nil {
		return "", err
	}
	if cfg.LogitSoftcap > 0 {
		scaled := prefix + "_logits_softcap_scaled"
		tanh := prefix + "_logits_softcap_tanh"
		capped := prefix + "_logits_softcapped"
		prog.ScalarMul(logits, 1.0/cfg.LogitSoftcap, scaled)
		prog.Tanh(scaled, tanh)
		prog.ScalarMul(tanh, cfg.LogitSoftcap, capped)
		logits = capped
	}
	targets := prefix + "_targets"
	spanMask := prefix + "_span_mask"
	loss := prefix + "_loss"
	acc := prefix + "_accuracy"
	cleanMean := prefix + "_clean_score_mean"
	corruptMean := prefix + "_corrupt_score_mean"
	scores := prefix + "_scores"
	prog.Slice("targets", rowStart, rowStart+rows, 1, 0, targets)
	prog.Slice("energy_span_mask", rowStart, rowStart+rows, 1, 0, spanMask)
	prog.SpanPLLPairwiseLoss(logits, targets, spanMask, seqLen, cfg.Training.MinimalPair.LossKind(), float32(cfg.Training.MinimalPair.Margin), loss, acc, cleanMean, corruptMean)
	prog.SpanPLLPool(logits, targets, spanMask, seqLen, scores)
	prog.DeclareOutput(loss, TensorFloat32, []int{1})
	prog.DeclareOutput(acc, TensorFloat32, []int{1})
	prog.DeclareOutput(cleanMean, TensorFloat32, []int{1})
	prog.DeclareOutput(corruptMean, TensorFloat32, []int{1})
	prog.DeclareOutput(scores, TensorFloat32, []int{rows / seqLen, 1})
	return loss, nil
}

func multiheadResolvedBlocks(blocks []BlockSpec) []BlockSpec {
	out := make([]BlockSpec, len(blocks))
	copy(out, blocks)
	for i := range out {
		if blockTypeKey(out[i]) == "plain" {
			out[i].AttentionMask = AttentionMaskBlockDiffusion
		}
	}
	return out
}

func multiheadAnyDWAHead(t TrainingSpec) bool {
	for _, head := range t.Heads {
		if head.LayerAggregation == LayerAggregationDWA {
			return true
		}
	}
	return false
}

func emitMultiheadHeadLogitsIR(prog *Program, prefix string, head MultiheadHeadSpec, hidden string, wi int, eps float32, dropout float32) (string, int, error) {
	switch head.OutputHead {
	case MultiheadOutputBERTMLM:
		ln1 := prefix + "_mlm_ln1"
		denseMM := prefix + "_mlm_dense_mm"
		dense := prefix + "_mlm_dense_out"
		gelu := prefix + "_mlm_gelu"
		ln2 := prefix + "_mlm_ln2"
		drop := prefix + "_mlm_dropout"
		tied := prefix + "_mlm_tied_weight"
		mm := prefix + "_logits_mm"
		logits := prefix + "_logits_raw"
		prog.LayerNormNoAffine(hidden, ln1, eps)
		prog.MatMul(ln1, weightName(wi), denseMM)
		wi++
		prog.Add(denseMM, weightName(wi), dense)
		wi++
		prog.GELU(dense, gelu)
		prog.LayerNormNoAffine(gelu, ln2, eps)
		headInput := ln2
		if dropout > 0 {
			prog.Dropout(headInput, dropout, drop)
			headInput = drop
		}
		prog.Transpose(weightName(0), []int{1, 0}, tied)
		prog.MatMul(headInput, tied, mm)
		prog.Add(mm, weightName(wi), logits)
		wi++
		return logits, wi, nil
	case MultiheadOutputLinear:
		logits := prefix + "_logits_raw"
		if head.TieEmbeddings {
			tied := prefix + "_tied_weight"
			prog.Transpose(weightName(0), []int{1, 0}, tied)
			prog.MatMul(hidden, tied, logits)
			return logits, wi, nil
		}
		prog.MatMul(hidden, weightName(wi), logits)
		return logits, wi + 1, nil
	case MultiheadOutputBinary:
		mm := prefix + "_binary_mm"
		logits := prefix + "_logits_raw"
		prog.MatMul(hidden, weightName(wi), mm)
		wi++
		prog.Add(mm, weightName(wi), logits)
		wi++
		return logits, wi, nil
	case MultiheadOutputScalar:
		mm := prefix + "_energy_mm"
		logits := prefix + "_logits_raw"
		prog.MatMul(hidden, weightName(wi), mm)
		wi++
		prog.Add(mm, weightName(wi), logits)
		wi++
		return logits, wi, nil
	default:
		return "", wi, fmt.Errorf("unsupported multihead output_head=%q", head.OutputHead)
	}
}
