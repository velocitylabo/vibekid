# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # SFT 100 件 eval — self-contained (HF Hub direct load) — #163 / C-4
#
# 目的: vibekid SFT (Phase 4) を **HF Hub から直接 load** + 100 件 eval set で deterministic generate。
# 出力 JSON をローカル `node sft/scripts/validate-results.mjs` (Playwright) で execute → exec_success_rate 計算。
#
# **Self-contained**: 訓練不要、Drive / Kaggle Dataset セットアップ不要。
# Notebook 1 file で完結 (eval prompts + system prompt は inline embed)。
#
# **環境**: Colab (推奨、T4/A100 どちらでも) or Kaggle T4 で動く。
# wall clock: T4 で ~50min、A100 で ~15-20min (100 件 × ~30s/prompt or ~10s/prompt)。
#
# **#163 C-4 経路詳細**:
# - Web 経路 (LiteRT) は Gemma 4 LoRA 公式未対応 (#129) のため Notebook 経路で代替計測
# - validate ロジックは `sft/eval-validate-runner.html` で `sft/eval-runner.html` (Web baseline 計測) と同一実装
# - system prompt は production 経路と同じ **app-oneshot** (`sft/eval-runner.html` SYSTEM_PROMPTS['app-oneshot'] と一致)
# - eval set 100 件は `sft/data/eval.jsonl` と一致 (本 notebook に inline embed、DRY 違反だが self-contained 優先)

# %% [markdown]
# ## 1. dependencies install (Phase 4 SFT と同じ pin)

# %%
# !pip install -qU unsloth transformers==5.5.0 peft trl bitsandbytes

# %% [markdown]
# ## 2. HF Hub から adapter direct load
#
# memory `project_external_research_findings.md` Section 8 経路。training を skip して
# adapter (Phase 4 で push 済) を直接 load、~30-60s で inference 準備完了。

# %%
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="velocitylabo/vibekid-gemma-4-E2B-lora-phase4",
    max_seq_length=2048,
    load_in_4bit=True,
)
print("loaded adapter: velocitylabo/vibekid-gemma-4-E2B-lora-phase4")

# %% [markdown]
# ## 3a. system prompt (app-oneshot、production 経路と一致)
#
# `sft/eval-runner.html` SYSTEM_PROMPTS['app-oneshot'] と同一文字列。
# writeup Section A/B の "SFT + oneshot" narrative と整合する production 経路の system prompt。

# %%
APP_ONESHOT_SYSTEM_PROMPT = """子供が「〜作って」と言ったら、p5.js のスケッチコードを1つだけ書いてください。

例（「ボールが跳ねる」の場合）:
```js
let ballX = 200;
let ballY = 100;
let ballVY = 0;

function setup() {
  createCanvas(400, 400);
}

function draw() {
  background(240);
  ballVY = ballVY + 0.5;
  ballY = ballY + ballVY;
  if (ballY > 370) {
    ballY = 370;
    ballVY = -10;
  }
  fill(255, 100, 100);
  noStroke();
  ellipse(ballX, ballY, 50, 50);
}
```

ルール:
- 変数は function の外で let で宣言する
- 関数は必ず setup() と draw() を書く
- createCanvas(400, 400) を使う
- background() を draw の最初に呼ぶ
- 変数名は ballX, bgColor, myScore のような分かりやすい名前を使う
- p5.js の組み込み関数名（color, fill, background, width, height）を変数名にしない
- 短く、動くコードだけ書く（20〜40行程度）
- ```js で囲んで出力する

同じスタイルで、指示されたものを作ってください:"""

# %% [markdown]
# ## 3b. eval prompts inline (talk-sample sft/data/eval.jsonl と一致、100 件)

# %%
import json

EVAL_PROMPTS_JSONL = """{"prompt":"ゆびでモノクロのらくがきをするやつで、まちがえたらやりなおしできるのがいいな！","category":"drawing","difficulty":"medium"}
{"prompt":"ネオンみたいにひかるまんだらもようをかいて、けしたい！","category":"drawing","difficulty":"medium"}
{"prompt":"カラフルなクイズでせいかいがおおきくなるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"あいさつがおおきくなって、へるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"ひよこがぴかぴかゆれるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"ぴかぴかふうせんが、ぴょんぴょんはねるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"にじいろのドアが、マウスについてくるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"キラキラのさかなが、わるいものからにげるゲームつくって！","category":"game","difficulty":"hard"}
{"prompt":"くるくるまわるちいさなおばけ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"どうぶつをあつめるゲームつくって！ドキドキするやつ！","category":"game","difficulty":"medium"}
{"prompt":"かわいいカレンダーで、ランダムにひづけがかわるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"パステルカラーの、おほしさまがマウスについてくるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"もぐらをたたいてやっつけるゲームをつくって！カラフルなやつ！","category":"game","difficulty":"medium"}
{"prompt":"なまえをかくと、もじがへっていくやつ、つくって","category":"ui-state","difficulty":"medium"}
{"prompt":"うちゅうじんからにげるゲームつくって！わくわくするやつがいいな！","category":"game","difficulty":"hard"}
{"prompt":"りんごをタップするとシュワシュワうごくやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"ふわふわまいおちる、おほしさまをつくって","category":"animation","difficulty":"medium"}
{"prompt":"クイズのボタンをおすと、こたえをえらべるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"じゃんけんゲームで、まるいライフがへっていくやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"かっこいいとけい、おしたらじかんがきりかわるやつ、つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"ゆびでモノクロのかたちをかくやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"ゆびでモノクロのスタンプをおしたい！","category":"drawing","difficulty":"easy"}
{"prompt":"ゆびでカラフルなてんてんをかくやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"うちゅうじんをたくさんあつめるゲームをつくって！カラフルなやつがいいな！","category":"game","difficulty":"hard"}
{"prompt":"おばけからにげるゲームをつくって！ドキドキするやつ！","category":"game","difficulty":"hard"}
{"prompt":"どうぶつさんを３つそろえるのんびりゲーム、つくって！","category":"game","difficulty":"medium"}
{"prompt":"パステルカラーのあめが、ふわふわとんでいくやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"うちゅうじんをたたくゲームで、のんびりしたやつ、つくって！","category":"game","difficulty":"hard"}
{"prompt":"レトロなゲームで、フルーツをあつめるやつ、つくって！へんなフルーツにあたるとダメなやつ！","category":"game","difficulty":"hard"}
{"prompt":"ゆびでカラフルなまるをかけるやつで、やりなおしできるのがいいな！","category":"drawing","difficulty":"medium"}
{"prompt":"ひこうきがくるくるながれるやつ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"タップするとふうせんがピカッとうごくやつ","category":"interaction","difficulty":"medium"}
{"prompt":"はやくさかなをみつけるゲームつくって！","category":"game","difficulty":"medium"}
{"prompt":"はらはらドキドキするパズルゲームをつくって！レベルがあがっていくやつがいいな！","category":"game","difficulty":"medium"}
{"prompt":"じゃんけんのぐーちょきぱーを、カラフルにボタンおしていれかえられるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"まるいボタンをおすと、ランダムなクイズがでるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"ゆびでなぞるとキラキラのおえかきができるやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"カラフルなクレヨンで、マウスでらくがきしたい！","category":"drawing","difficulty":"medium"}
{"prompt":"マウスをおすとネオンみたいにひかるせんのスタンプがおせるやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"きょうはなんにちかな？カレンダーをリセットできるやつ、つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"プレゼントをおすと、ネオンみたいにピカッとひかるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"くるくるまわるあめが、いろをかえながらおちてくるやつ、つくって！","category":"animation","difficulty":"medium"}
{"prompt":"にじいろのしゃぼんだまがいっぱいとぶやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"おすとキラキラゆきがふるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"ゆきだるまさんがくるくるはしるやつ、つくって！","category":"animation","difficulty":"medium"}
{"prompt":"マウスをおすと、キラキラのはなもようがスタンプみたいにいっぱいできるやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"カラフルなさかなをみつけるゲームつくって！はやくみつけたいな！","category":"game","difficulty":"hard"}
{"prompt":"キラキラおつきさまが、ながれていくやつ、つくって！","category":"animation","difficulty":"medium"}
{"prompt":"カラフルなクイズゲームで、こたえをえらべるやつ、つくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"ゆびで、にじいろのかたちをえらんで、おえかきしたい！","category":"drawing","difficulty":"medium"}
{"prompt":"マウスをおすと、カラフルなランダムもようのてんてんがいっぱいかくやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"キラキラもぐらをたくさんあつめるゲームつくって！","category":"game","difficulty":"medium"}
{"prompt":"あめをおすと、どんどんおおきくなるカラフルなやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"あいさつがおすとでてきて、リセットできるかわいいやつ、つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"パステルカラーのスタンプを、マウスでポンポンおせるやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"ボタンをおすとネオンみたいにきえるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"レトロなゲームで、むしをよけるやつ、タイマーがあるやつ、つくって！","category":"game","difficulty":"hard"}
{"prompt":"ゆびでカラフルなせんをかくやつで、けせるのがいいな！","category":"drawing","difficulty":"medium"}
{"prompt":"ドアをおすと、ポンポンきえちゃうやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"クレヨンで まんだらもようを スタンプしたいな！","category":"drawing","difficulty":"medium"}
{"prompt":"おおきいタイマーが、リセットできるやつ、つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"クレヨンでてんてんのランダムもようをかくやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"スピードがはやいカードからにげるゲームつくって！","category":"game","difficulty":"hard"}
{"prompt":"カラフルなまるをかくやつ、つくって！けすこともできるやつがいいな！","category":"drawing","difficulty":"medium"}
{"prompt":"パステルカラーのつきが、ふわふわまいおちるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"タップすると、パステルカラーのハートがふわふわうごくやつ、つくって！","category":"interaction","difficulty":"easy"}
{"prompt":"ぴかぴかのおさかなが、おおきくなるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"レトロなゲームで、もぐらをたたくやつ、つくって！","category":"game","difficulty":"hard"}
{"prompt":"そらからぴかぴかほしがふるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"マウスがおいつくとドアがにじいろになるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"あめをおすと、ネオンみたいにピカッとおとがなるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"かっこいい とけいが リセットできる やつ つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"おほしさまがクリックでポンポンふえるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"キラキラのパズルゲームつくって！スコアがふえるやつ！","category":"game","difficulty":"hard"}
{"prompt":"なまえがおされるじゅんばんでかわるやつ、カラフルにつくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"たまごを おすと、パッと ひかるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"しんごうきの色をえらべるやつ、カラフルにつくって！","category":"ui-state","difficulty":"medium"}
{"prompt":"ポップなしんごうきをボタンでおしたいな！","category":"ui-state","difficulty":"medium"}
{"prompt":"ボタンをおすと、ピカッてひかるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"ふわふわ いぬさんが いろかわるやつ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"にじいろのたいようがながれていくやつ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"キラキラのくもが、だんだんおおきくなるやつ、つくって！","category":"animation","difficulty":"easy"}
{"prompt":"おほしさまをクリックすると、キラキラがいっぱいふえるやつ、つくって！","category":"interaction","difficulty":"easy"}
{"prompt":"どうぶつをおすと、いろんなのがでてくるやつ、つくって！カラフルだと嬉しいな！","category":"interaction","difficulty":"medium"}
{"prompt":"にじいろでらくがきしたい！いろもえらべるやつ、つくって！","category":"drawing","difficulty":"medium"}
{"prompt":"ぴかぴかのおばけが、まいおちるやつ、つくって！","category":"animation","difficulty":"medium"}
{"prompt":"まるいカレンダー、ひづけをいれかえたいな！","category":"ui-state","difficulty":"medium"}
{"prompt":"ドアをおすと、パステルカラーの音がなるやつ、つくって！","category":"interaction","difficulty":"medium"}
{"prompt":"フルーツがおいかけてくるドキドキゲームつくって！","category":"game","difficulty":"hard"}
{"prompt":"ボールがあたるとゆっくりかわるゲームつくって","category":"game","difficulty":"medium"}
{"prompt":"ハートをおすと、ドカーンってきえちゃうやつ、つくって！","category":"interaction","difficulty":"easy"}
{"prompt":"ゆきがゆらゆらふるやつ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"カラフルなもようをランダムでいっぱいつくって！","category":"drawing","difficulty":"medium"}
{"prompt":"カラフルなあいさつのことばを、きりかえたいな！","category":"ui-state","difficulty":"medium"}
{"prompt":"かわいいスコアがふえるやつ、つくって！","category":"ui-state","difficulty":"easy"}
{"prompt":"ぬりえをマウスでカラフルにかきたいな！","category":"drawing","difficulty":"medium"}
{"prompt":"パステルカラーのもこもこくもがふわふわゆれるやつ、つくって！","category":"animation","difficulty":"medium"}
{"prompt":"くるくるねこさんがきえたりでたりするやつ、つくって","category":"animation","difficulty":"easy"}
{"prompt":"キラキラのなかを、とりさんがにげるゲームつくって！","category":"game","difficulty":"hard"}
{"prompt":"あめがポンポンおちてきて、タップするとうごくやつつくって","category":"interaction","difficulty":"medium"}
"""

EVAL_PROMPTS = [json.loads(line) for line in EVAL_PROMPTS_JSONL.strip().split("\n") if line.strip()]
print(f"loaded {len(EVAL_PROMPTS)} eval prompts (inline embed)")
assert len(EVAL_PROMPTS) == 100, f"expected 100 prompts, got {len(EVAL_PROMPTS)}"

# %% [markdown]
# ## 4. 100 件 deterministic generate

# %%
import time

results = []
t_start = time.time()
for idx, item in enumerate(EVAL_PROMPTS):
    prompt_text = item["prompt"]
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": f"{APP_ONESHOT_SYSTEM_PROMPT}\n\n{prompt_text}"}],
    }]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    t0 = time.time()
    out = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=False,  # deterministic (memory feedback_gemma_determinism.md)
    )
    gen_secs = time.time() - t0
    raw_text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    results.append({
        "idx": idx,
        "prompt": prompt_text,
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "raw_text": raw_text,
        "gen_secs": gen_secs,
    })
    if (idx + 1) % 10 == 0:
        elapsed = time.time() - t_start
        est_total = elapsed * len(EVAL_PROMPTS) / (idx + 1)
        print(f"  [{idx + 1}/{len(EVAL_PROMPTS)}] elapsed={elapsed:.1f}s, est_total={est_total:.1f}s")

mean_gen = sum(r["gen_secs"] for r in results) / max(1, len(results))
print(f"\ngeneration done: n={len(results)}, mean_gen_secs={mean_gen:.2f}s")

# %% [markdown]
# ## 5. JSON dump
#
# 出力先 `eval_phase4_sft.json`:
# - Colab: `/content/eval_phase4_sft.json` (DL 必要)
# - Kaggle: `/kaggle/working/eval_phase4_sft.json` (output で取得)
# - local: `./eval_phase4_sft.json`

# %%
import os

if os.path.exists("/kaggle/working"):
    OUT = "/kaggle/working/eval_phase4_sft.json"
elif os.path.exists("/content"):
    OUT = "/content/eval_phase4_sft.json"
else:
    OUT = "./eval_phase4_sft.json"

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "velocitylabo/vibekid-gemma-4-E2B-lora-phase4",
        "system_prompt": "app-oneshot (eval-runner.html SYSTEM_PROMPTS['app-oneshot'])",
        "max_new_tokens": 1024,
        "do_sample": False,
        "n": len(results),
        "mean_gen_secs": mean_gen,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"[saved] {OUT}")
print(f"  n={len(results)}, mean_gen_secs={mean_gen:.2f}s")
print(f"\n次:")
print(f"  1. {OUT} をローカルに DL → sft/logs/eval_phase4_sft.json に配置")
print(f"  2. node sft/scripts/validate-results.mjs --in=sft/logs/eval_phase4_sft.json")
print(f"  3. SFT_SCORE 確定 → HF model card placeholder 埋め (#163 / C-4 Phase C)")
