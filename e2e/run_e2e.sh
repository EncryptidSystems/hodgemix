#!/bin/bash
# run_e2e.sh DIR [PORT]: serve DIR/model.bnz (+ .sig, model.bnz.ternary.safetensors, config.json) with bnz-loader's
# bnz4serve and open www/e2e.html in headless Chrome (SwiftShader WebGPU = CPU). Signed with BL/testkeys/test.seed.
# Needs: bnz-loader (BL), edgellm/bitnet/kernels (KERN), google-chrome; www/ needs symlinks to
# BL/js/{bnz-loader,bnz4dec,bitnet-rt,shaders}.js and BL/js/vendor.
set -u; cd "$(dirname "$0")"; D=$1; P=${2:-8123}
BL=${BL:-/workspace/integration/tgz/bnz-loader}; KERN=${KERN:-/workspace/edgellm/bitnet/kernels}
for f in bnz-loader.js bnz4dec.js bitnet-rt.js shaders.js vendor; do [ -e www/$f ] || ln -s $BL/js/$f www/$f; done
timeout 280 $BL/bnz4serve -addr 127.0.0.1:$P -js www -kernels $KERN -config $D/config.json -bnz $D/model.bnz \
  -st $D/model.bnz.ternary.safetensors -pubkey $BL/testkeys/test.pub > $D/e2e.log 2>&1 & SRV=$!
sleep 0.5
PROF=$(mktemp -d)
timeout 270 google-chrome --headless=new --no-sandbox --enable-unsafe-webgpu --enable-features=Vulkan --use-webgpu-adapter=swiftshader \
  --user-data-dir="$PROF" --no-first-run "http://127.0.0.1:$P/e2e.html" > $D/chrome.log 2>&1 & CHR=$!
wait $SRV; kill $CHR 2>/dev/null; sleep 1; pkill -f "$PROF" 2>/dev/null; rm -rf "$PROF"
grep -a -E "RESULT|FAIL|GPU error" $D/e2e.log
