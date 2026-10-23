package checkedint

import (
	"math"
	"strings"
	"testing"
)

func TestProduct(t *testing.T) {
	for _, c := range []struct {
		factors []int
		want    int
	}{
		{nil, 1},
		{[]int{7}, 7},
		{[]int{3, 4, 5}, 60},
		{[]int{0, math.MaxInt}, 0},
		{[]int{math.MaxInt, 0}, 0},
		{[]int{math.MaxInt, 1}, math.MaxInt},
		{[]int{math.MaxInt / 2, 2}, math.MaxInt - 1},
	} {
		got, err := Product(c.factors...)
		if err != nil || got != c.want {
			t.Errorf("Product(%v) = %d, %v; want %d", c.factors, got, err, c.want)
		}
	}
}

func TestProductRejects(t *testing.T) {
	for _, c := range []struct {
		factors []int
		want    string
	}{
		{[]int{math.MaxInt/2 + 1, 2}, "overflows"},
		{[]int{1 << 32, 1 << 32}, "overflows"},
		{[]int{2, 3, math.MaxInt / 5}, "overflows"},
		{[]int{4, -1}, "negative"},
		{[]int{0, -1}, "negative"},
	} {
		got, err := Product(c.factors...)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("Product(%v) = %d, %v; want error containing %q", c.factors, got, err, c.want)
		}
	}
}

func TestSum(t *testing.T) {
	for _, c := range []struct {
		terms []int
		want  int
	}{
		{nil, 0},
		{[]int{3, 4, 5}, 12},
		{[]int{math.MaxInt - 1, 1}, math.MaxInt},
	} {
		got, err := Sum(c.terms...)
		if err != nil || got != c.want {
			t.Errorf("Sum(%v) = %d, %v; want %d", c.terms, got, err, c.want)
		}
	}
}

func TestSumRejects(t *testing.T) {
	for _, c := range []struct {
		terms []int
		want  string
	}{
		{[]int{math.MaxInt, 1}, "overflows"},
		{[]int{math.MaxInt / 2, math.MaxInt / 2, 2}, "overflows"},
		{[]int{5, -1}, "negative"},
	} {
		got, err := Sum(c.terms...)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("Sum(%v) = %d, %v; want error containing %q", c.terms, got, err, c.want)
		}
	}
}
