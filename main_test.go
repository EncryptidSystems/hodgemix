package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestMain removes earlier runs' hodgemix-test-* folders first: a run's own cannot be removed while it
// still holds its model files open (Windows), so each run clears what the previous ones left.
func TestMain(m *testing.M) {
	if old, _ := filepath.Glob(filepath.Join(os.TempDir(), "hodgemix-test-*")); len(old) > 0 {
		for _, d := range old {
			_ = os.RemoveAll(d)
		}
	}
	os.Exit(m.Run())
}
