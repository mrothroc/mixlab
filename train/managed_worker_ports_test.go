package train

import (
	"fmt"
	"net"
	"testing"
	"time"
)

func managedRingPorts(t *testing.T) []int {
	t.Helper()
	// Avoid the ephemeral client range: a connecting rank can otherwise claim
	// the next rank's listener port before that rank reaches bind().
	for range 30 {
		port := 20000 + int(time.Now().UnixNano()%9999)
		first, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			continue
		}
		second, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", port+1))
		_ = first.Close()
		if err != nil {
			continue
		}
		_ = second.Close()
		return []int{port, port + 1}
	}
	t.Fatal("could not reserve local ring ports")
	return nil
}
