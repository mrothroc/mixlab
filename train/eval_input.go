package train

import (
	"fmt"
	"path/filepath"
	"strings"
)

type evalShardSelection struct {
	Pattern  string
	Explicit bool
}

func resolveEvalShardPattern(trainPattern, valPattern string) (evalShardSelection, error) {
	selection := evalShardSelection{Pattern: valPattern, Explicit: valPattern != ""}
	if !selection.Explicit {
		if trainPattern == "" {
			return evalShardSelection{}, fmt.Errorf("evaluation data is required; pass -val '<glob>' (preferred) or -train '<glob>' for legacy val-pattern derivation")
		}
		selection.Pattern = strings.Replace(trainPattern, "train", "val", 1)
	}
	matches, err := filepath.Glob(selection.Pattern)
	if err != nil {
		flag := "-val"
		if !selection.Explicit {
			flag = "evaluation glob derived from -train"
		}
		return evalShardSelection{}, fmt.Errorf("invalid %s pattern %q: %w", flag, selection.Pattern, err)
	}
	if len(matches) == 0 {
		if selection.Explicit {
			return evalShardSelection{}, fmt.Errorf("-val pattern %q matched no shard files", selection.Pattern)
		}
		return evalShardSelection{}, fmt.Errorf(
			"evaluation glob %q derived from -train %q matched no shard files; pass -val '<glob>' to evaluate an explicit split (including train_* shards prepared with -val-split 0)",
			selection.Pattern, trainPattern,
		)
	}
	return selection, nil
}

func (s evalShardSelection) sourceLabel() string {
	if s.Explicit {
		return "explicit -val"
	}
	return "derived from -train"
}
