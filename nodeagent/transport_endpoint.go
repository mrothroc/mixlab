package nodeagent

import (
	"fmt"
	"net"
	"strconv"
)

// ValidateTransportEndpoint defines the administrator's advertised bind policy.
// Job plans must match it exactly; discovery cannot choose another local IP.
func ValidateTransportEndpoint(address string) error {
	host, port, err := net.SplitHostPort(address)
	ip := net.ParseIP(host)
	n, numberErr := strconv.Atoi(port)
	if err != nil || numberErr != nil || ip == nil || ip.IsUnspecified() || ip.IsMulticast() || n < 1024 || n > 65535 || net.JoinHostPort(ip.String(), strconv.Itoa(n)) != address {
		return fmt.Errorf("explicit canonical local relay IP and port required")
	}
	return nil
}
