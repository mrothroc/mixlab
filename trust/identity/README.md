# Public Identity Phrases

`mixlab_wordlist_v1` is the unchanged 2,048-word English BIP-39 vocabulary,
pinned at bitcoin/bips commit `ce1862ac6bcffa1dd20aad858380e51e66e949ea`:
<https://github.com/bitcoin/bips/blob/ce1862ac6bcffa1dd20aad858380e51e66e949ea/bip-0039/english.txt>.

Its SHA-256 is
`2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda`.
Tests lock the ordering, uniqueness, checksum and bit convention. Changing the
vocabulary requires a new protocol version, not editing this file in place.

Only the vocabulary is reused: these are not BIP-39 wallet mnemonics, checksums,
or seed derivations. Eight words render the first 88 bits of a cluster's public
root-SPKI fingerprint, most-significant bit first. The full fingerprint is
always returned alongside. Five words render the first 55 bits of a separately
computed confirmation digest; enrollment must bind that digest to its channel,
request and protocol domain. Phrases alone confer no authority.

Source authors: Marek Palatinus, Pavol Rusnak, ThomasV, Aaron Voisine, Sean Bowe.
BIP-39 is MIT licensed; see `THIRD_PARTY_NOTICES.md` at the repository root.
