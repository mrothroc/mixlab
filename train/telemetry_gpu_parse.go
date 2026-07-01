package train

import (
	"regexp"
	"strconv"
)

var ioregGPUUtilRE = regexp.MustCompile(`"Device Utilization %"\s*=\s*([0-9]+(?:\.[0-9]+)?)`)

func parseIORegGPUUtilPercent(out string) *float64 {
	matches := ioregGPUUtilRE.FindAllStringSubmatch(out, -1)
	if len(matches) == 0 {
		return nil
	}
	var maxUtil float64
	found := false
	for _, m := range matches {
		if len(m) < 2 {
			continue
		}
		v, err := strconv.ParseFloat(m[1], 64)
		if err != nil {
			continue
		}
		if !found || v > maxUtil {
			maxUtil = v
			found = true
		}
	}
	if !found {
		return nil
	}
	return &maxUtil
}
