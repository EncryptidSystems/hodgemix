package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] == "desk" {
		if err := serve(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	var err error
	switch os.Args[1] {
	case "strip":
		err = stripCLI(os.Args[2:])
	case "text":
		err = textCLI(os.Args[2:])
	case "extract":
		err = extractCLI(os.Args[2:])
	case "corpus":
		err = corpusCLI(os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, "itt desk -dir DIR | itt strip -dir DIR | itt text -cochain FILE -tokenizer FILE -text TEXT | itt extract -cochain FILE -tokenizer FILE -text TEXT | itt corpus -cochain FILE -tokenizer FILE -dir DIR")
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func stripCLI(args []string) error {
	fs := flag.NewFlagSet("strip", flag.ExitOnError)
	dir := fs.String("dir", "", "instrumented model folder")
	out := fs.String("out", "", "integer cochain safetensors (default DIR/model.cochain.safetensors)")
	fs.Parse(args)
	if *dir == "" {
		return fmt.Errorf("strip: -dir is required")
	}
	rows, cells, err := StripDir(*dir, *out)
	if err != nil {
		return err
	}
	dest := *out
	if dest == "" {
		dest = *dir + string(os.PathSeparator) + "model.cochain.safetensors"
	}
	fmt.Printf("%d rows × %d cells → %s\n", rows, cells, dest)
	return nil
}

func textCLI(args []string) error {
	fs := flag.NewFlagSet("text", flag.ExitOnError)
	co := fs.String("cochain", "", "integer cochain safetensors")
	tok := fs.String("tokenizer", "", "tokenizer.json")
	cfg := fs.String("config", "", "tokenizer_config.json (optional)")
	text := fs.String("text", "", "the text")
	out := fs.String("out", "", "safetensors for the anchor chain and the top cochain")
	fs.Parse(args)
	if *co == "" || *tok == "" || *text == "" {
		return fmt.Errorf("text: -cochain, -tokenizer, and -text are required")
	}
	res, err := Text(*co, *tok, *cfg, *text, *out)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(res)
}

func extractCLI(args []string) error {
	fs := flag.NewFlagSet("extract", flag.ExitOnError)
	co := fs.String("cochain", "", "integer cochain safetensors")
	tok := fs.String("tokenizer", "", "tokenizer.json")
	cfg := fs.String("config", "", "tokenizer_config.json (optional)")
	vol := fs.String("volume", "", "volume.safetensors (default: beside the cochain)")
	text := fs.String("text", "", "the text")
	out := fs.String("out", "", "float weight safetensors (default: beside the cochain, text.weight.safetensors)")
	fs.Parse(args)
	if *co == "" || *tok == "" || *text == "" {
		return fmt.Errorf("extract: -cochain, -tokenizer, and -text are required")
	}
	ex, err := Extract(*co, *tok, *cfg, *vol, *text, *out)
	if err != nil {
		return err
	}
	kind := "cells"
	if ex.Lifted {
		kind = "channels"
	}
	fmt.Printf("%d words × %d %s → %s\n", ex.Words, ex.Dim, kind, ex.Path)
	return nil
}

func corpusCLI(args []string) error {
	fs := flag.NewFlagSet("corpus", flag.ExitOnError)
	co := fs.String("cochain", "", "integer cochain safetensors")
	tok := fs.String("tokenizer", "", "tokenizer.json")
	cfg := fs.String("config", "", "tokenizer_config.json (optional)")
	dir := fs.String("dir", "", "folder of .txt works, one file per text")
	out := fs.String("out", "", "float weight safetensors, one row per text")
	fs.Parse(args)
	if *co == "" || *tok == "" || *dir == "" {
		return fmt.Errorf("corpus: -cochain, -tokenizer, and -dir are required")
	}
	works, err := readWorks(*dir)
	if err != nil {
		return err
	}
	m, err := Corpus(*co, *tok, *cfg, works, *out)
	if err != nil {
		return err
	}
	for i, w := range m.Works {
		fmt.Printf("%s  %s  %d words\n", w.Name, w.SHA256, w.Words)
		if i < len(m.Edges) {
			e := m.Edges[i]
			next := m.Works[e.To].Name
			if e.Same {
				fmt.Printf("  same cochain as %s\n", next)
			} else {
				fmt.Printf("  differs from %s  L1 %g\n", next, e.L1)
			}
		}
	}
	if m.Path != "" {
		fmt.Printf("%d texts × %d cells → %s\n", len(m.Works), m.Cells, m.Path)
	}
	return nil
}

func readWorks(dir string) ([]Work, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if strings.EqualFold(filepath.Ext(e.Name()), ".txt") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return nil, fmt.Errorf("%s has no .txt files", dir)
	}
	var works []Work
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		works = append(works, Work{Name: name, Text: string(b)})
	}
	return works, nil
}
