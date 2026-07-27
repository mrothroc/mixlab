//go:build !mlx || !cgo || (!darwin && !linux)

package gpu

func mlxGroupRuntimeCreate(string, bool) int64 { return 0 }
func mlxGroupRuntimeRank(int64) int            { return -1 }
func mlxGroupRuntimeWorldSize(int64) int       { return -1 }
func mlxGroupRuntimeValidateIdentity(
	int64,
	uint64,
	[8]uint32,
	[]uint32,
	[8]uint32,
) int {
	return -1
}
func mlxGroupRuntimeValidateManifest(int64, []uint64) (int, int, int) {
	return -1, -1, -1
}
func mlxGroupRuntimeBroadcastControl(int64, int, []int32) ([]int32, int) {
	return nil, -1
}
func mlxGroupRuntimeDestroy(int64) {}
