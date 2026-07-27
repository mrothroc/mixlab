package train

func fillDropoutKeys(dst []int32, seed uint64, step int) {
	fillDistributedDropoutKeys(dst, seed, step, 0, 0)
}

func fillDistributedDropoutKeys(
	dst []int32,
	seed uint64,
	step, rank, microstep int,
) {
	for i := 0; i+1 < len(dst); i += 2 {
		ordinal := uint64(i / 2)
		x := seed ^
			(uint64(step)+1)*0x9e3779b97f4a7c15 ^
			(uint64(rank)+1)*0x94d049bb133111eb ^
			(uint64(microstep)+1)*0xbf58476d1ce4e5b9 ^
			(ordinal+1)*0xd1b54a32d192ed03
		if rank == 0 && microstep == 0 {
			// Preserve the pre-distributed singleton key sequence.
			x = seed ^ (uint64(step)+1)*0x9e3779b97f4a7c15 ^
				(ordinal+1)*0xd1b54a32d192ed03
		}
		a := splitMix64(x)
		b := splitMix64(a)
		dst[i] = int32(uint32(a))
		dst[i+1] = int32(uint32(b))
	}
}
