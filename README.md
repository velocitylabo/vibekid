# VibeKid — On-device Vibe Coding for Kids

> Say "ねこがはしる" ("a cat runs") in Japanese, and p5.js code is
> generated and the picture starts moving — AI and code both stay inside
> the browser, with no server traffic.

A submission for the Kaggle Gemma 4 Hackathon (2026). The demo video is
distributed via the Kaggle submission writeup.

## What

- An **on-device LLM** (Gemma 4 E2B, LiteRT-LM + WebGPU) generates p5.js
  code from a child's words.
- Generated code is rendered immediately inside an iframe; we confirm it
  is running via a heartbeat.
- A PoC aimed at letting **5–7-year-olds turn the picture they want to
  describe into a moving picture**.
- Everything stays inside the Chrome browser (**no network traffic;
  privacy preserved**).

## Why

When a child thinks "I want to draw a picture" or "I want to make a ball
move," they should be able to express it in their own words, without the
hurdle of learning to code. Because it runs on-device:

- No internet connection required (works even if home Wi-Fi is down).
- Personal information never leaves for a server (peace of mind for
  parents).
- No AI usage fees (easy on the household budget).

## Quick Start

### Operating environment

- **Latest Chrome** (verify WebGPU support in DevTools: `chrome://gpu`,
  Backend = D3D12 recommended).
- **Windows 11 + dGPU recommended** (NVIDIA Turing generation or newer;
  `shader-f16` extension required).
- ⚠️ WSL2 is not supported (the dzn driver does not deliver practical
  speed).
- Linux native + Vulkan dGPU + `shader-f16` also works
  (environment-dependent).
- Mac depends on the MoltenVK environment.

Details: see the "Operating environment" section of
[`docs/sync-vibekid.md`](./docs/sync-vibekid.md).

### Model download (first time only)

`models/gemma-4-E2B-it-web.task` (~3 GB, LiteRT format) is required.
Obtain it from Google's official
[`litert-community/gemma-4-E2B-it-web`](https://huggingface.co/litert-community/gemma-4-E2B-it-web):

```bash
# First-time download (consent to the Gemma Terms of Use on HF Hub is
# required)
mkdir -p models
wget -O models/gemma-4-E2B-it-web.task \
  https://huggingface.co/litert-community/gemma-4-E2B-it-web/resolve/main/gemma-4-E2B-it-web.task
```

The vibekid app also has a built-in path to fetch from the same URL on
first launch and cache it in OPFS (`MODEL_URL` in `app.js`); from the
second launch onward it boots instantly on a cache hit.

### Launch

```bash
git clone https://github.com/velocitylabo/vibekid.git
cd vibekid
python3 -m http.server 8000
# Open http://localhost:8000 in Chrome
```

First launch: model download (~3 GB) → OPFS cache (instant boot from the
next time) → welcome screen.

Press any of the five emoji presets (🐱🏀🔘☔ + the animal / physics /
interactive / visual categories), or type something freely like
"○○ がはねる" ("~ bounces"), and the generate → preview cycle completes
in about 5 seconds.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Chrome (client-only; no server required)                         │
│                                                                  │
│  Alpine.js v3 (UI)  ──►  LiteRT-LM (Gemma 4 E2B)  ──►  p5.js   │
│         ↑                  ↑              ↓             ↓       │
│      user input       OPFS cache    generated code   iframe     │
│                                                       preview   │
└─────────────────────────────────────────────────────────────────┘
```

- **UI**: Alpine.js v3 (CDN, no build step, 3 files: `index.html` /
  `app.js` / `style.css`).
- **AI**: LiteRT-LM via `@mediapipe/tasks-genai`, Gemma 4 E2B (`.task`
  format, ~3 GB, WebGPU inference).
- **Model persistence**: stores the initial download in OPFS (Origin
  Private File System); instant on a cache hit at restart.
- **Generation validation**: runs the generated p5.js code in an iframe
  sandbox; deems it "working" when a heartbeat is received (judged OK
  after 2.5 s of continuous heartbeat).

## Highlights

### 🎯 Offline operation + personal-information protection

You can verify this from network logs (DevTools Network tab). Aside from
the initial model download, there is no server traffic at runtime.

### 🧠 Gemma 4 E2B + p5.js-specific LoRA fine-tuning

A 757-item child-targeted p5.js synthesis dataset was synthesized via
Gemini 2.5 Flash and used to QLoRA fine-tune Gemma 4 E2B. **The LoRA
adapters are published on HF Hub**:

- [`velocitylabo/vibekid-gemma-4-E2B-lora-phase4`](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4)
  — Phase 4 LoRA (cap rate 14%, `train_on_responses_only` + Gemma 4
  end-of-turn token fix).
- [`velocitylabo/vibekid-gemma-4-E2B-lora`](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora)
  — Phase 3 archive (cap rate 43%, kept for comparison).

The specific landmines we stepped on (`merged_4bit_forced`
NotImplementedError / T4 not supporting bf16 / `chat_template.jinja` not
bundled / silent breakage of `mm_token_type_ids`) and the workarounds are
documented in detail in the HF Hub model card. Published as a
contribution to the Unsloth community.

### 🎬 Child N=2 user testing

Operational footage and reaction cuts from actual 5–7-year-olds using the
tool. Incorporated into the submission writeup as qualitative observation
(not as a claim of statistical significance with N=2, but as a hands-on
look at the UX).

### 📊 Reproducibility

- **HF Hub LoRA adapter** + model card (baseline / SFT eval numbers).
- **Kaggle Notebook** (1-click reproduction of the base vs SFT ablation
  on T4).

https://www.kaggle.com/code/pirozhiki/vibekid-kaggle-bench-for-gemma-4-e2b-lora

## Known limitations

- **Prompt scope**: optimized for generating child-targeted Japanese
  p5.js global-mode code; general-purpose JS / multilingual is not
  guaranteed.
- **Base-model limits**:
  - Drawings of concrete objects (cat / dog / person) tend to be
    substituted with primitives (a base-model trait; expected to be
    mitigated by SFT).
  - Event handlers (`mousePressed`, `keyPressed`) tend to be replaced
    with per-frame checks inside `draw()`.
  - The model card / Notebook present bench numbers showing that these
    are mitigated after SFT.
- **Operating environment**: an environment that satisfies WebGPU +
  shader-f16 + maxBufferSize ≥ 1.5 GB (Chrome native is recommended).
- **Model size**: requires 3 GB on first run; hard in low-bandwidth /
  low-storage environments.

## Directory layout

```
vibekid/
├── index.html / app.js / style.css   # The Web app itself (3 files)
├── icons/ vendor/                    # Assets
├── tools/
│   ├── cdp-sampling.mjs              # Bench automation via CDP (5 presets × N runs)
│   └── filter-repo-to-vibekid.sh     # Sync script from the dev repo (submission prep)
├── sft/                              # SFT pipeline
│   ├── scripts/                      # Synthesis / evaluation / Modal merge
│   ├── notebook/                     # Colab SFT + Kaggle bench
│   ├── logs/                         # Eval baseline JSON
│   ├── model-card/README.md          # Model card for HF Hub
│   └── seeds.jsonl + meta-prompt.template.md
└── docs/
    ├── sync-vibekid.md               # Dev / submission sync procedure
    └── modal-merge-runbook.md        # Modal merge operating procedure
```

## Development history

This repo carries a **path-filtered git history** from the upstream
development repository, so the development trace is visible commit by
commit (`git log --oneline`). It contains roughly a month of trial and
error, from `refactor: 雑談機能を全削除` ("remove all small-talk
functionality") to `chore(tools): CDP 経由の N runs サンプリング自動化`
("automate N-run sampling via CDP").

## License

- **Code / Docs**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (see LICENSE file; consistent with the Kaggle Gemma 4 Hackathon
  submission license requirements).
- **Model (base)**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
  — complies with the Gemma 4 license.
- **Model (LoRA adapter)**: inherits the Gemma Terms as a derivative of
  the base model `google/gemma-4-E2B-it`; the HF Hub model-card text is
  CC-BY 4.0 dual-licensed.
- **Training Data**: synthetic data; because it includes Gemini 2.5
  Flash output, it complies with the OpenRouter / Google terms of use.

## Acknowledgements

- **Google DeepMind**: Gemma 4 + LiteRT-LM.
- **Unsloth AI**: SFT toolkit + community.
- [p5.js](https://p5js.org/) / [Alpine.js](https://alpinejs.dev/).
- Hosts of the Kaggle Gemma 4 Hackathon.

## Citation

```bibtex
@misc{vibekid2026,
  title  = {VibeKid: On-device Vibe Coding PoC for Japanese Children},
  author = {Murakami, Hiroshi and {velocitylabo contributors}},
  year   = {2026},
  note   = {Kaggle Gemma 4 Hackathon},
  url    = {https://github.com/velocitylabo/vibekid}
}
```
