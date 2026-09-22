package train

import "fmt"

func weightInitLinearFanIn(ws WeightShape, policy string) int {
	// Architecture-specific state/gate initializers take precedence.
	if ws.InitMode != "" || ws.InitLogArange || ws.InitDtBias || ws.IsNormScale || ws.IsBuffer || ws.InitOne || ws.InitValue != 0 {
		return 0
	}
	if policy == "pytorch_linear_all" && ws.LinearFanIn > 0 {
		return ws.LinearFanIn
	}
	if policy == "pytorch_linear" || policy == "pytorch_linear_all" {
		return ws.PyTorchLinearFanIn
	}
	return 0
}

func weightInitCoverage(shapes []WeightShape, policy string) string {
	if policy != "pytorch_linear" && policy != "pytorch_linear_all" {
		return ""
	}
	matrices, biases, appliedMatrices, appliedBiases, excluded, overrides := 0, 0, 0, 0, 0, 0
	for _, ws := range shapes {
		if ws.NormalInitStd != nil {
			overrides++
		}
		if weightInitLinearFanIn(ws, "pytorch_linear_all") == 0 {
			excluded++
			continue
		}
		applied := weightInitLinearFanIn(ws, policy) > 0
		if len(ws.Shape) == 1 {
			biases++
			if applied {
				appliedBiases++
			}
		} else {
			matrices++
			if applied {
				appliedMatrices++
			}
		}
	}
	return fmt.Sprintf("weight_init: %s applied to %d/%d ordinary affine matrices and %d/%d paired biases; %d matrices retain Xavier, %d tensors retain separate policies (%d explicit normal overrides)",
		policy, appliedMatrices, matrices, appliedBiases, biases, matrices-appliedMatrices, excluded, overrides)
}
