# hodgemix

hodgemix builds one ternary model from a base checkpoint and one or more open-weight contributors. The mix is Hodge–Kuramoto alignment in weight space: seed alignment, a Hodge split, ternary TIES on the curl, and a selective DTFT lock on the harmonic part. There is no loss, no backward pass, and no training step.

The output is a BitNet b1.58 layout in BNZ4, the container [Champ](https://ai.encryptid.systems) and bnz-loader read. Audio and vision towers on a contributor are carried into that file. They are not folded into the text mix.

Encryptid Systems. The method is the paper *Ternary Conversion and Mixing of Open-Weight Models via Hodge–Kuramoto Alignment* (`paper_v2.pdf` in this tree). Binding choices are in `DECISIONS.md`. Open items are in `GAPS.md`.

Weights are not in this repository. A run reads the checkpoints named in its arguments or in a run file.

## Build

Go 1.24.

```
go build -o hodgemix.exe .
go build -o spm2json.exe ./spm2json
go build -o stdiff.exe ./stdiff
```

```
go test ./...
hodgemix --selftest
```

`hodgemix --dtft-tests` runs the DTFT lock checks on small synthetic grids.

## Run

One base and one contributor:

```
hodgemix --base BASE.safetensors --src CONTRIBUTOR.safetensors --out OUT.bnz
```

A second contributor is another `--src`. A weight other than 1 is `--src PATH:WEIGHT`. The base weight is `--base-weight` (default 1).

The usual way to name every file is a run file. Paths may be absolute, or relative to the run file. A named file that is missing stops the run. Nothing is looked up beside a checkpoint.

```yaml
base:
  model: models/base/model.safetensors   # or a list of shard paths
  tokenizer: models/base/tokenizer.json
  tokenizer_config: models/base/tokenizer_config.json
  config: models/base/config.json
  special_tokens_map: models/base/special_tokens_map.json   # optional
  generation_config: models/base/generation_config.json     # optional
sources:
  - model: models/contributor/model.safetensors
    weight: 1
    tokenizer: models/contributor/tokenizer.json
    config: models/contributor/config.json
out: out/hodge.bnz
settings:          # optional flag names
  limbs: 1
```

```
hodgemix --config RUN.yaml
```

`settings` accepts the flag names below (`limbs`, `rotate`, `topology`, and the rest). Unknown keys stop the run. A finished run writes `OUT.yaml` beside the BNZ, with the output's own files filled in. Probe commands take that file:

```
hodgemix --config OUT.yaml --score OUT.ternary.safetensors
hodgemix --config OUT.yaml --chat OUT.ternary.safetensors
hodgemix --config OUT.yaml --generate OUT.ternary.safetensors "The capital of France is" 8
```

Files under `runs/` are operator run files for this machine. They are not a portable example.

## What a run writes

| File | Contents |
|---|---|
| `OUT.bnz` | BNZ4 container. Methods 0, 1, 4, and 5. JSON index. Per-segment SHA-256. |
| `OUT.bnz.sig` | Ed25519 signature over the header and index, when `--sign-seed` is set. |
| `OUT.ternary.safetensors` | Packed ternary weights, scales, and channel-map tensors. Kept. |
| `OUT.report.json` | The run record, including `champ_compatible`. |
| `OUT.yaml` | Run file for the output, for later probes. |
| `volume.safetensors` | Per-role cell volumes on the married axes. |
| `config.json`, tokenizer files | Written beside the output from the files the run named. |

`--out-merged-st` also writes the full-precision tensors after rotation and before ternarization. `--no-verify` skips the BNZ round trip.

`weight_scale` is the mean absolute value over nonzero codes. A tensor the base already has keeps the base magnitude. Champ multiplies by that scale.

## Scale fix

An earlier writer stored the new tensor's own absmean as `weight_scale`. On an already-ternary tensor that value is the code scale times the fraction of nonzero codes, and a contributor's absmean belongs to that contributor's norm gains. On the first 2B4T + Qwen3-0.6B run those scales sat at about 0.27–0.39× the base on every BitLinear, and base codes with those scales moved perplexity on the fixed sentences from 18 to 60. The stored scale is now the mean absolute weight over the nonzero codes. Absmean stays the RoundClip threshold. An unchanged tensor gets the base scale back.

## End-to-end

`e2e/run_e2e.sh DIR` serves `DIR/model.bnz` with bnz-loader's `bnz4serve` and opens `e2e/www/e2e.html` in headless Chrome. The page uses bnz-loader's JavaScript unchanged. The check is: the signature verifies, every segment decoded by BNZ4Dec matches the ternary safetensors bytes, and the runtime weight-buffer digest and logits match a reference safetensors upload. `--bnz-force-rans` forces methods 4 and 5 on a tiny file. The recorded ternmix-py run (`--rotate off`, three sources) was 79/79 segments bit-exact, with and without `--bnz-force-rans`. `--rotate on` throws "untied LM head" in that runtime, which is the C8 refusal.

## Defaults

Every constant comes from `paper_v2` Appendix A unless a flag overrides it (`params.go`, decision 10).

| Flag | Default | Meaning |
|---|---|---|
| `--rotate` | `off` | Hadamard residual rotation. `on` unties the LM head, which Champ refuses. |
| `--lock` | `dtft` | Selective DTFT lock (`dtft_lock.pdf`, Algorithm D). `v2` is the earlier ω = 0 lock. |
| `--limbs` | `1` | Body digits. `1` is the ternary code loaders already read. |
| `--limb-tol` | `0.15` | Stop adding body limbs once the residual is below this fraction of the weight norm. |
| `--topology` | `base` | Target sizes. `max` takes the larger axis. `married` takes the lcm, so each model's radix hierarchy embeds. |
| `--index` | `married` | One mixed-radix heap per axis, shared by every contributor. `own` is a separate index per model. |
| `--mrh` | `dynamic` | Heap radices from each axis. `paper` is the 16-bucket heap in `paper_v2`. |
| `--vocab` | `union` | Base token ids stay. Other contributor tokens are appended. `base` keeps the base vocabulary. |
| `--attn` | `on` | Fold QK-norm, marry RoPE frequencies, match logit temperature to the base. |
| `--ov` | `chart` | Keep the base V and carry its change into O. `mix` mixes both factors. |
| `--seed-align` | `auto` (`on`) | Phase-correlation shift, then Re-Basin assignment, per tensor. |
| `--pin-mean` | `on` | A mixed matrix keeps the base response on the base forward's mean input. |
| `--surgery` | `on` | Base codes return on the singular set of the base forward. |
| `--repair` | `on` | RMSNorm gains so each channel's spread matches the base on the calibration tokens. |
| `--base-residual` | `on` | Keep base detail finer than the index. |
| `--float-dtype` | `bf16` | Storage for embeddings, norms, and other non-ternary tensors. |
| `--eval` | `on` | Score the base and the output after the write. |

`--rotate on` is refused by preflight: Champ loads a tied head only. Use it only for a runtime that accepts an untied head.

A run stops when a step cannot do what it describes. Preflight checks configs against the tensors, tokenizer ids against the embedding rows, and that no text tensor is left without a role. Calibration and the union vocabulary run before the mix. The Champ check runs before compression.

## Audio and vision

A contributor that nests its text stack under `model.language_model` (Gemma 4) still has its audio tower and vision tower in the file. Those tensors, their embedding projections, and the per-layer input embeddings are copied into the output under the same names. They are not text BitLinear roles.

The mix also writes least-squares tower readouts as `bnzmix.vision_tower`, `bnzmix.audio_tower`, `bnzmix.embed_vision`, and `bnzmix.embed_audio`. The text mix still writes `model.embed_tokens.weight`. Dropping a named tower is a scope cut and has to be stated. It is not a default.

## Probes

| Command | What it does |
|---|---|
| `--score MODEL [tokenizer.json]` | Greedy lines and negative log likelihood on the fixed sentences. |
| `--chat MODEL [tokenizer.json]` | One long reply in the page chat format. |
| `--generate MODEL "text" [n]` | `n` new tokens (default 8), with one manifold line per token. |
| `--converse MODEL [tokenizer.json]` | Twelve conversations. Transcript in `<model>.converse.json`. |
| `--trace BASE OUT [tokenizer.json]` | Residual cosine and the channels that moved, sublayer by sublayer. |
| `--flow MODEL OUT.safetensors "text"` | Hodge split of the forward cochain. Needs `--config`. |
| `--bisect` | Mix each role alone, score it, write `OUT.bisect.json`, delete the role files. |
| `--map BASE SRC` | Fit of the shared-vocabulary residual map. |
| `--ov-chart BASE OUT.ternary.safetensors` | Apply the OV chart to an existing output. |
| `--surgery BASE OUT.ternary.safetensors [tokenizer.json]` | Surgery in place. |
| `--repair BASE OUT.ternary.safetensors [tokenizer.json]` | Repair in place. |
| `--neck TERNARY OUT` | Body-limb pack without remixing. Needs `--config`. Llama 3 rope scaling only. |
| `--sing PACK OUT` | Append neck limbs to an older pack. Needs `--config`. |

`--update UPDATE` on `--generate`, `--chat`, or `--flow` keeps that file's feed-forward as a second expert on the same cells. A token uses it only when the combined block still moves the base residual cell the same way.

## Other tools

`spm2json` writes the tokenizer.json hodgemix and Champ read from a SentencePiece BPE `tokenizer.model`:

```
spm2json -model tokenizer.model -config tokenizer_config.json -out tokenizer.json
```

`stdiff` compares two safetensors files tensor by tensor, including packed ternary codes:

```
stdiff -a BASE.safetensors -b OUT.safetensors -json report.json
```

## Recorded comparisons

These are runs already made. They are not a promise about a new pair of checkpoints.

- BitNet b1.58 2B4T on the six fixed sentences scores perplexity 12.0. The greedy continuation of "The capital of France is" begins " Paris. Paris is a city that is known for".
- 2B4T + Qwen3-0.6B at 1:1, base topology: the curl-set TIES mean used to ignore `--base-weight` and replace the base's curl set (perplexity 7.6×10^7). Counting every contributor's weight, then seed alignment, surgery, pin-mean, and the OV chart, brought that run to perplexity 14.4. The OV cross term was 0.23 of the base circuit norm.
- The same pair with `--limbs 2` scored perplexity 4142. Stock 2B4T moved entirely into a radix-4 limb still scores 12.0, so the limb arithmetic is exact and the extra limb was keeping a residual the shared-vocabulary map does not align. The default stays one body limb.
- `stdiff` of hodge-2b4t-1 against Microsoft's 2B4T: 542 of 542 tensors byte-identical, 0 of 2,084,044,800 codes changed, plus the `bnzmix` channel-map tensors.
- Hodge-3.5 (`hodge-vaultgemma-1`) assimilates VaultGemma into the 2B4T width. Hodge-4 assimilates Gemma 4 into Hodge-3.5 and keeps the audio and vision towers. Its reported perplexity is 1.06× Hodge-3.5. `stdiff` against the source checkpoints found no byte-identical tensor in either pair.

Channel matching is still per tensor. A source row is not tied to the base row it adds into. That limit is `GAPS.md` item 18.

## Neuromorphic event path

`eval/neuro.go` is two exact forms of an int8 BitLinear code. A nonzero code is one graded spike. The non-adjacent form writes the same integer as digits in {−1, 0, +1}. Both match the integer accumulator of the LUT fabric, so the scaled output matches today's forward bit for bit. `kernels/neuro.wgsl` compacts the nonzero codes and accumulates them on WebGPU, with the same packing, centre, scale, and `ACCUM` as `bitlinear.wgsl`.

Champ still dispatches `bitlinear.wgsl`. The event kernel is not on that path. The rest of `neuromorphic.md` is a design note. It was not measured on a checkpoint.

## Where to read next

| Document | What it is |
|---|---|
| `paper_v2.pdf`, `paper_v2.tex` | The mix. Equation numbers in the code comments refer to it. |
| `dtft_lock.pdf` | The default lock. |
| `TRACE.md` | Equation, function, and test. |
| `DECISIONS.md` | Choices that the code is required to follow, including Champ C1–C10. |
| `GAPS.md` | What is still open. |
| `fec.md` | Bounded commuting projection on the married-axis grid. |
| `neuromorphic.md` | Event-path design. The graded kernel is the part that is built. |

`tower/` is the audio and vision tower math. `bnz4/` is the BNZ4 codec. `eval/` is the CPU reference forward, including the LUT fabric.

## Copyright

Copyright Encryptid Systems. This repository does not grant an open-source license.
