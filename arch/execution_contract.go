package arch

import "encoding/json"

type ExecutionProfile string

const (
	NativeFull              ExecutionProfile = "native-full"
	NativeTTTStateful       ExecutionProfile = "native-ttt-stateful"
	ExecutionContractFormat                  = "mixlab.execution_contract.v1"
)

// ContractError is suitable for machine-readable CLI diagnostics.
type ContractError struct {
	Code   string `json:"code"`
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

func (e *ContractError) Error() string              { return e.Code + " at " + e.Path + ": " + e.Reason }
func contractError(code, path, reason string) error { return &ContractError{code, path, reason} }

type ContractAxis struct {
	Name string `json:"name"`
	Size int    `json:"size"`
}
type ContractTensor struct {
	DType   int            `json:"dtype"`
	Shape   []int          `json:"shape"`
	Axes    []ContractAxis `json:"axes"`
	Packing [][]string     `json:"packing"`
}
type ContractRepresentation struct {
	ID        string         `json:"id"`
	Binding   string         `json:"binding"`
	Direction string         `json:"direction"`
	Meaning   string         `json:"meaning"`
	Tensor    ContractTensor `json:"tensor"`
}
type ContractParameter struct {
	ID        string `json:"id"`
	Index     int    `json:"index"`
	Name      string `json:"name"`
	GroupID   string `json:"parameter_group_id"`
	Shape     []int  `json:"shape"`
	Frozen    bool   `json:"frozen"`
	Buffer    bool   `json:"buffer"`
	Reachable bool   `json:"reachable"`
}
type ContractInitSlice struct {
	ParameterID      string `json:"parameter_id"`
	SourceStart      int    `json:"source_start"`
	DestinationStart int    `json:"destination_start"`
	Length           int    `json:"length"`
}
type ContractInitialization struct {
	Kind   string              `json:"kind"`
	Slices []ContractInitSlice `json:"slices,omitempty"`
	Value  string              `json:"value,omitempty"`
}
type ContractControl struct {
	Type         string `json:"type"`
	Initial      int    `json:"initial"`
	Min          int    `json:"min"`
	MaxExclusive int    `json:"max_exclusive"`
}
type ContractState struct {
	ID             string                 `json:"id"`
	OwnerID        string                 `json:"owner_id"`
	Role           string                 `json:"role"`
	Storage        string                 `json:"storage"`
	Tensor         *ContractTensor        `json:"tensor,omitempty"`
	Control        *ContractControl       `json:"control,omitempty"`
	Binding        string                 `json:"binding"`
	Input          string                 `json:"input,omitempty"`
	Output         string                 `json:"output,omitempty"`
	Initialization ContractInitialization `json:"initialization"`
	Lifetime       string                 `json:"lifetime"`
	Reset          string                 `json:"reset"`
	Gradient       string                 `json:"gradient"`
	Growth         string                 `json:"growth"`
	SchemaVersion  int                    `json:"schema_version"`
	Compatibility  []string               `json:"compatibility"`
}
type ContractBoundary struct {
	ID              string   `json:"id"`
	OwnerID         string   `json:"owner_id"`
	Stages          []string `json:"stages"`
	Reads           []string `json:"reads"`
	Writes          []string `json:"writes"`
	Callable        bool     `json:"independently_callable"`
	ObservationUnit string   `json:"observation_unit"`
	Ordering        string   `json:"ordering,omitempty"`
	InputValidity   string   `json:"input_validity,omitempty"`
	UpdateFrequency string   `json:"update_frequency,omitempty"`
	OutputTiming    string   `json:"output_timing,omitempty"`
	Commit          string   `json:"commit_boundary"`
	Description     string   `json:"description"`
}
type ContractCapability struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Reason string `json:"reason"`
}
type ContractMemory struct {
	Kind   string `json:"kind"`
	Bytes  *int64 `json:"bytes"`
	Reason string `json:"reason"`
}
type ContractOpBinding struct {
	Code        int       `json:"code"`
	Inputs      []string  `json:"inputs"`
	Outputs     []string  `json:"outputs"`
	IntParams   []int     `json:"int_params"`
	FloatParams []float32 `json:"float_params"`
}

func contractOpBinding(op Op) ContractOpBinding {
	return ContractOpBinding{Code: op.Code, Inputs: append([]string(nil), op.Inputs...), Outputs: append([]string(nil), op.Outputs...), IntParams: append([]int(nil), op.IntParams...), FloatParams: append([]float32(nil), op.FloatParams...)}
}

// ProgramContract holds bindings to the actual emitted graph, not a second IR.
type ProgramContract struct {
	Representations []ContractRepresentation `json:"representations"`
	States          []ContractState          `json:"states"`
	Ops             []ContractOpBinding      `json:"ops"`
	SelectedOutput  string                   `json:"selected_output,omitempty"`
	DetachedEdges   []ContractEdge           `json:"detached_edges,omitempty"`
}
type ContractEdge struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}
type BlockContract struct {
	BlockID          string              `json:"block_id"`
	OwnerID          string              `json:"owner_id"`
	ParameterGroupID string              `json:"parameter_group_id"`
	Kind             string              `json:"kind"`
	Described        bool                `json:"described"`
	ParameterIndices []int               `json:"parameter_indices"`
	States           []ContractState     `json:"states"`
	Boundaries       []ContractBoundary  `json:"boundaries"`
	Bindings         []ContractOpBinding `json:"bindings"`
}
type ContractContext struct {
	ModelDim, SeqLen, BatchSize, VocabSize int
	BlockIndex, SiteIndex, WeightIndex     int
	Stream                                 string
	Profile                                ExecutionProfile
	ComputeDType                           string
}
type BlockContractProvider func(BlockSpec, ContractContext) (BlockContract, error)
type ExecutionRequest struct {
	Capability string
	ShareState bool
}
type ModelContract struct {
	Format             string               `json:"format"`
	SchemaVersion      int                  `json:"schema_version"`
	Profile            ExecutionProfile     `json:"profile"`
	ComputeDType       string               `json:"compute_dtype"`
	ConfigDigest       string               `json:"effective_config_digest"`
	SchemaDigest       string               `json:"schema_digest"`
	SourceRevision     string               `json:"source_revision"`
	CheckpointIdentity *string              `json:"checkpoint_identity"`
	Coverage           string               `json:"coverage"`
	Blocks             []BlockContract      `json:"blocks"`
	Parameters         []ContractParameter  `json:"parameters"`
	Program            ProgramContract      `json:"program"`
	Capabilities       []ContractCapability `json:"capabilities"`
	Memory             []ContractMemory     `json:"memory"`
	Diagnostics        []ContractError      `json:"diagnostics"`
	Lifecycle          *ContractLifecycle   `json:"lifecycle,omitempty"`
}

type ContractLifecycle struct {
	StateOwnership  string   `json:"state_ownership"`
	Concurrency     string   `json:"concurrency"`
	ResetFailure    string   `json:"reset_failure"`
	Close           string   `json:"close"`
	OutputRetention []string `json:"output_retention"`
}

// Clone returns an independent host-only snapshot, including nested slices.
func (c *ModelContract) Clone() *ModelContract {
	if c == nil {
		return nil
	}
	b, err := json.Marshal(c)
	if err != nil {
		panic(err)
	}
	var out ModelContract
	if err := json.Unmarshal(b, &out); err != nil {
		panic(err)
	}
	return &out
}
