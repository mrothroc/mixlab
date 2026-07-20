package train

import "fmt"

func logDatasetFraming(cfg *ArchConfig, name string, batchSize int) {
	if cfg.Training.ExampleFramingEnabled() {
		f := cfg.Training.ExampleFraming
		fmt.Printf("  [%s] example framing: content_len=%d bos_id=%d eos_id=%d examples/batch=%d\n",
			name, f.ContentLen, f.BosID, f.EosID, batchSize)
	}
	if cfg.Training.RecordFramingEnabled() {
		fmt.Printf("  [%s] per-record framing: bos_id=%d eos_id=%d pad_id=%d examples/batch=%d\n",
			name, cfg.Training.DatasetBOSID, cfg.Training.DatasetEOSID, cfg.Training.DatasetPADID, batchSize)
	}
}
