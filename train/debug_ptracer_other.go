//go:build !linux

package train

func allowDebugParent() error { return nil }
