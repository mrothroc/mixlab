package train

import (
	"os"
	"path/filepath"
	"sort"
)

// tokenizerPathForTrainPattern finds tokenizer metadata adjacent to the first
// matched shard. Keeping this lookup shared prevents tokenizer-backed training
// features from drifting to different artifact discovery rules.
func tokenizerPathForTrainPattern(trainPattern string) (string, bool, error) {
	matches, err := filepath.Glob(trainPattern)
	if err != nil {
		return "", false, err
	}
	sort.Strings(matches)
	if len(matches) > 0 {
		path := filepath.Join(filepath.Dir(matches[0]), "tokenizer.json")
		if _, err := os.Stat(path); err == nil {
			return path, true, nil
		}
	}
	path := filepath.Join(filepath.Dir(trainPattern), "tokenizer.json")
	if _, err := os.Stat(path); err == nil {
		return path, true, nil
	}
	return "", false, nil
}
