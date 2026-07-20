package train

func fillDropoutKeys(dst []int32, seed uint64, step int) {
	for i := 0; i+1 < len(dst); i += 2 {
		ordinal := uint64(i / 2)
		x := seed ^ (uint64(step)+1)*0x9e3779b97f4a7c15 ^ (ordinal+1)*0xd1b54a32d192ed03
		a := splitMix64(x)
		b := splitMix64(a)
		dst[i] = int32(uint32(a))
		dst[i+1] = int32(uint32(b))
	}
}

func splitMix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
