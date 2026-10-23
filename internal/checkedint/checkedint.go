// Package checkedint computes buffer sizes from configuration values and file
// headers without silently wrapping, so an absurd shape fails with an error
// instead of allocating a wrapped (short) buffer.
package checkedint

import (
	"fmt"
	"math"
)

// Product returns the product of non-negative factors. It fails on a negative
// factor or when the product exceeds math.MaxInt.
func Product(factors ...int) (int, error) {
	product := 1
	for _, f := range factors {
		if f < 0 {
			return 0, fmt.Errorf("negative size factor %d in %v", f, factors)
		}
	}
	for _, f := range factors {
		if f == 0 {
			return 0, nil
		}
		if product > math.MaxInt/f {
			return 0, fmt.Errorf("size product %v overflows int", factors)
		}
		product *= f
	}
	return product, nil
}

// Sum returns the sum of non-negative terms. It fails on a negative term or
// when the sum exceeds math.MaxInt.
func Sum(terms ...int) (int, error) {
	sum := 0
	for _, t := range terms {
		if t < 0 {
			return 0, fmt.Errorf("negative size term %d in %v", t, terms)
		}
		if sum > math.MaxInt-t {
			return 0, fmt.Errorf("size sum %v overflows int", terms)
		}
		sum += t
	}
	return sum, nil
}
