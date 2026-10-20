//go:build !windows

package bnz4

import (
	"os"
	"syscall"
)

func mmapImpl(path string) ([]byte, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, _ := f.Stat()
	if st.Size() == 0 {
		return []byte{}, func() {}, nil
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return b, func() { syscall.Munmap(b) }, nil
}
