//go:build windows

package bnz4

import "os"

// Windows: plain read (no syscall.Mmap). Upstream candidate for bitnet-compress/wgsl/bnz.
func mmapImpl(path string) ([]byte, func(), error) {
	b, err := os.ReadFile(path)
	return b, func() {}, err
}
