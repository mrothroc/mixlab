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
	B := rawBatch * headCount
	T := cfg.SeqLen
	D := cfg.ModelDim
	V := cfg.VocabSize
	rawRows := rawBatch * T
	nWeightsMeta, err := collectMultiheadWeightShapesFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	nWeights := len(nWeightsMeta)
	prog := NewProgram(nWeights)
	prog.DeclareInput("tokens", TensorInt32, []int{B, T})
	prog.DeclareInput("targets", TensorInt32, []int{B * T})
	prog.DeclareInput("loss_mask", TensorFloat32, []int{B * T})
	prog.DeclareInput("diffusion_block_start", TensorInt32, []int{B})
	prog.DeclareInput("diffusion_block_end", TensorInt32, []int{B})
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
	if wi != blockWeightStart+len(blockShapes) {
		return nil, fmt.Errorf("multihead trunk weight count mismatch: emitted=%d expected=%d", wi-blockWeightStart, len(blockShapes))
	}
	wi += adalnWeights

	exportHeadName := cfg.Training.ExportHead
	diffusionHeadName := cfg.Training.DiffusionHead
	lossAccum := ""
	norm := cfg.EffectiveNormSpec()
	for headIdx, head := range cfg.Training.Heads {
		rowStart := headIdx * rawRows
		headPrefix := "head_" + head.Name
		headIn := headPrefix + "_hidden"
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
		targets := headPrefix + "_targets"
		mask := headPrefix + "_loss_mask"
		loss := headPrefix + "_loss"
		weighted := headPrefix + "_loss_weighted"
		prog.Slice("targets", rowStart, rowStart+rawRows, 1, 0, targets)
		prog.Slice("loss_mask", rowStart, rowStart+rawRows, 1, 0, mask)
		lossWeight := float32(head.LossWeight)
		if head.Objective == ObjectiveRTD {
			prog.MaskedBCEWithLogits(logits, targets, mask, loss)
			acc := headPrefix + "_accuracy"
			prog.MaskedBinaryAccuracy(logits, targets, mask, acc)
			prog.DeclareOutput(acc, TensorFloat32, []int{1})
			if cfg.Training.RTD != nil {
				lossWeight *= float32(cfg.Training.RTD.DiscriminatorLossWeight)
			}
		} else {
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
		if head.Name == exportHeadName || head.Name == diffusionHeadName || head.Name == generatorOutputName {
			prog.ScalarMul(logits, 1.0, headPrefix+"_logits")
			outCols := V
			if head.Objective == ObjectiveRTD {
				outCols = 1
			}
			prog.DeclareOutput(headPrefix+"_logits", TensorFloat32, []int{rawRows, outCols})
		} else if head.Objective == ObjectiveRTD {
			prog.ScalarMul(logits, 1.0, headPrefix+"_logits")
			prog.DeclareOutput(headPrefix+"_logits", TensorFloat32, []int{rawRows, 1})
		}
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
	prog.Slice("x", 0, rawRows, 1, 0, "x_hidden_flat")
	prog.Reshape("x_hidden_flat", []int{rawBatch, T, D}, "x_hidden")
	prog.DeclareOutput("x_hidden", TensorFloat32, []int{rawBatch, T, D})
	if wi != nWeights {
		return nil, fmt.Errorf("multihead IR weight count mismatch: emitted=%d expected=%d", wi, nWeights)
	}
	return prog, nil
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
	default:
		return "", wi, fmt.Errorf("unsupported multihead output_head=%q", head.OutputHead)
	}
}
