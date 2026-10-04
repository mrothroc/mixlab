package train

import (
	"encoding/json"
	"errors"
	"io"

	"github.com/mrothroc/mixlab/arch"
)

// RunInspectContract is deliberately independent of GPU/runtime setup.
// It writes one JSON report only after all requested checks have succeeded.
func RunInspectContract(configPath, profile, require string, out io.Writer) error {
	if configPath == "" {
		return &arch.ContractError{Code: "invalid_config", Path: "config", Reason: "-config is required"}
	}
	cfg, err := arch.LoadArchConfig(configPath)
	if err != nil {
		return &arch.ContractError{Code: "invalid_config", Path: "config", Reason: err.Error()}
	}
	c, err := arch.DescribeModelContract(cfg, arch.ExecutionProfile(profile))
	if err != nil {
		return err
	}
	if require != "" {
		if err := arch.RequireExecution(c, arch.ExecutionRequest{Capability: require}); err != nil {
			return err
		}
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(c)
}

// WriteContractDiagnostic keeps machine-readable errors off the JSON stdout.
func WriteContractDiagnostic(out io.Writer, err error) error {
	var detail *arch.ContractError
	if !errors.As(err, &detail) {
		detail = &arch.ContractError{Code: "inspection_failed", Path: "inspect-contract", Reason: err.Error()}
	}
	return json.NewEncoder(out).Encode(detail)
}
