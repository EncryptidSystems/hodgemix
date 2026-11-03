package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

//go:embed desk.html
var deskFS embed.FS

// audit is the model this deployment mounts. The browser never sends a path.
type audit struct {
	cochain string
	tok     string
	cfg     string
	volume  string
}

func serve() error {
	fs := flag.NewFlagSet("desk", flag.ContinueOnError)
	dir := fs.String("dir", os.Getenv("ITT_MODEL"), "model folder mounted for this deployment")
	addr := fs.String("addr", "127.0.0.1:58164", "listen address")
	if err := fs.Parse(deskArgs()); err != nil {
		return err
	}
	a, err := openAudit(*dir)
	if err != nil {
		return err
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		b, err := deskFS.ReadFile("desk.html")
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(b)
	})
	mux.HandleFunc("/api/text", post(a.textAPI))
	mux.HandleFunc("/api/corpus", post(a.corpusAPI))
	mux.HandleFunc("/api/volume", a.volumeFile)
	fmt.Println("http://" + ln.Addr().String())
	return http.Serve(ln, mux)
}

func deskArgs() []string {
	if len(os.Args) >= 2 && os.Args[1] == "desk" {
		return os.Args[2:]
	}
	return nil
}

func openAudit(dir string) (*audit, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, fmt.Errorf("desk: -dir or ITT_MODEL is the mounted model folder")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	vol := filepath.Join(abs, "volume.safetensors")
	if _, err := os.Stat(vol); err != nil {
		fmt.Fprintf(os.Stderr, "volume: %s\n", vol)
		return nil, fmt.Errorf("the model volume is not mounted")
	}
	tok := filepath.Join(abs, "tokenizer.json")
	if _, err := os.Stat(tok); err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %s\n", tok)
		return nil, fmt.Errorf("the model tokenizer is not mounted")
	}
	cfg := filepath.Join(abs, "tokenizer_config.json")
	if _, err := os.Stat(cfg); err != nil {
		cfg = ""
	}
	co := filepath.Join(abs, "model.cochain.safetensors")
	if _, err := os.Stat(co); err != nil {
		if _, _, err := StripDir(abs, co); err != nil {
			fmt.Fprintf(os.Stderr, "cochain: %v\n", err)
			return nil, fmt.Errorf("the model cochain is not ready")
		}
	}
	return &audit{cochain: co, tok: tok, cfg: cfg, volume: vol}, nil
}

func post(fn func([]byte) (any, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, 32<<20))
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		out, err := fn(body)
		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		json.NewEncoder(w).Encode(out)
	}
}

func (a *audit) textAPI(body []byte) (any, error) {
	var req struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	return Text(a.cochain, a.tok, a.cfg, req.Text, "")
}

func (a *audit) corpusAPI(body []byte) (any, error) {
	var req struct {
		Works []Work `json:"works"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	return Corpus(a.cochain, a.tok, a.cfg, req.Works, "")
}

// volumeFile serves the instrumented volume mounted with this deployment.
func (a *audit) volumeFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET", http.StatusMethodNotAllowed)
		return
	}
	http.ServeFile(w, r, a.volume)
}
