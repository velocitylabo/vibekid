#!/usr/bin/env node
/**
 * SFT データ合成: スロット充填式 meta-prompt → Gemini API → validate → append
 *
 * Usage:
 *   # 単発（スロット明示）
 *   node build-prompt.mjs --category=game --subject=さかな --mechanic=スコア --style=ぴかぴか
 *
 *   # バッチ（グリッドから N 件ランダム抽出）
 *   node build-prompt.mjs --batch=10 [--category=animation]
 *
 *   # 共通オプション
 *   [--model=gemini-2.5-flash] [--out=data/sft.jsonl] [--raw-dir=data/raw] [--dry-run]
 *
 * Env: GEMINI_API_KEY
 */
import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parseArgs } from "util";
import { execFileSync } from "child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SFT_ROOT = join(__dirname, "..");

const { values: args } = parseArgs({
  options: {
    category:    { type: "string" },
    subject:     { type: "string" },
    mechanic:    { type: "string" },
    style:       { type: "string" },
    difficulty:  { type: "string" },
    batch:       { type: "string" },
    model:       { type: "string", default: "gemini-2.5-flash" },
    out:         { type: "string", default: join(SFT_ROOT, "data", "sft.jsonl") },
    "raw-dir":   { type: "string", default: join(SFT_ROOT, "data", "raw") },
    "dry-run":   { type: "boolean", default: false },
  },
});

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("GEMINI_API_KEY 環境変数が未設定");
  process.exit(1);
}

const MODEL = args.model;

// ============================================================
// スロット定義（組み合わせグリッド）
// ============================================================

const SLOTS = {
  animation: {
    subjects: [
      "ねこ", "いぬ", "うさぎ", "ひよこ", "くま", "ぺんぎん", "かめ", "ちょうちょ",
      "おさかな", "ほし", "つき", "たいよう", "くも", "にじ", "はな", "おちば",
      "ゆきだるま", "ロケット", "UFO", "でんしゃ", "ひこうき", "ふうせん",
      "しゃぼんだま", "あめ", "ゆき", "おばけ", "かぼちゃ", "さくら",
    ],
    mechanics: [
      "はしる", "とぶ", "おちる", "まわる", "はねる", "ゆれる", "ひかる",
      "おおきくなる", "ちいさくなる", "いろがかわる", "きえたりでたり",
      "ながれる", "ふる", "まいおちる", "のぼる",
    ],
    styles: [
      "にじいろ", "キラキラ", "ふわふわ", "ぴかぴか", "くるくる", "ぴょんぴょん",
      "ゆらゆら", "もこもこ", "カラフル", "パステル",
    ],
  },
  interaction: {
    subjects: [
      "ボタン", "はなび", "ふうせん", "どうぶつ", "あめ", "ゆき", "おほしさま",
      "りんご", "ハート", "おんぷ", "スイッチ", "ドア", "はこ", "マジック",
      "びっくりばこ", "たまご", "プレゼント", "でんき",
    ],
    mechanics: [
      "おすとでる", "クリックでふえる", "おすといろがかわる", "タップでうごく",
      "おすときえる", "おすとおおきくなる", "おすとおとがなる", "おすとひかる",
      "マウスでついてくる", "クリックでばくはつ",
    ],
    styles: [
      "キラキラ", "ドカーン", "パッ", "シュワシュワ", "ポンポン",
      "ピカッ", "にじいろ", "カラフル", "パステル", "ネオン",
    ],
  },
  "ui-state": {
    subjects: [
      "カウンター", "タイマー", "スコア", "とけい", "でんたく", "しんごうき",
      "サイコロ", "おみくじ", "じゃんけん", "クイズ", "なまえ", "あいさつ",
      "きぶんメーター", "おてんき", "カレンダー",
    ],
    mechanics: [
      "ふえる", "へる", "リセット", "きりかえ", "ランダム", "じゅんばん",
      "えらぶ", "いれかえ", "セーブ",
    ],
    styles: [
      "かわいい", "かっこいい", "シンプル", "カラフル", "パステル",
      "まるい", "おおきい", "ポップ",
    ],
  },
  game: {
    subjects: [
      "もぐら", "おばけ", "フルーツ", "ほし", "さかな", "むし", "とり",
      "うちゅうじん", "にんじゃ", "ボール", "ブロック", "カード", "パズル",
      "めいろ", "ヘビ", "どうぶつ",
    ],
    mechanics: [
      "たたく", "よける", "あつめる", "にげる", "みつける", "そろえる",
      "たおす", "ジャンプ", "スコア", "タイマー", "レベル", "あたりはんてい",
    ],
    styles: [
      "ドキドキ", "わくわく", "はらはら", "のんびり", "スピード",
      "キラキラ", "カラフル", "レトロ",
    ],
  },
  drawing: {
    subjects: [
      "おえかき", "ぬりえ", "スタンプ", "もよう", "まんだら", "かたち",
      "せん", "てん", "にじ", "はなもよう", "ドット", "らくがき",
    ],
    mechanics: [
      "ゆびでかく", "マウスでかく", "いろをえらぶ", "ふとさをかえる",
      "けす", "やりなおし", "スタンプおす", "ランダムもよう",
    ],
    styles: [
      "カラフル", "パステル", "にじいろ", "ネオン", "クレヨン",
      "すいさい", "キラキラ", "モノクロ",
    ],
  },
};

const DIFFICULTY_MAP = {
  animation: ["easy", "easy", "medium"],
  interaction: ["easy", "medium", "medium"],
  "ui-state": ["easy", "medium", "medium"],
  game: ["medium", "hard", "hard"],
  drawing: ["easy", "medium", "medium"],
};

// ============================================================
// ヘルパー
// ============================================================

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function pickDifficulty(category) {
  const pool = DIFFICULTY_MAP[category] || ["easy", "medium"];
  return pickRandom(pool);
}

function normalizePrompt(text) {
  return (text || "").normalize("NFKC").replace(/\s+/g, "");
}

/** 既存 prompt の正規化セットを返す */
function loadExistingNormalized(outPath) {
  const set = new Set();
  if (existsSync(outPath)) {
    const lines = readFileSync(outPath, "utf-8").trim().split("\n").filter(Boolean);
    for (const l of lines) set.add(normalizePrompt(JSON.parse(l).prompt));
  }
  // seeds も追加
  const seedPath = join(SFT_ROOT, "seeds.jsonl");
  if (existsSync(seedPath)) {
    const lines = readFileSync(seedPath, "utf-8").trim().split("\n").filter(Boolean);
    for (const l of lines) set.add(normalizePrompt(JSON.parse(l).prompt));
  }
  return set;
}

/** スロットから 1 件分の指定を生成（重複チェック付き） */
function generateSlot(category, existingNorm, maxRetries = 20) {
  const cat = category || pickRandom(Object.keys(SLOTS));
  const slots = SLOTS[cat];

  for (let i = 0; i < maxRetries; i++) {
    const subject = pickRandom(slots.subjects);
    const mechanic = pickRandom(slots.mechanics);
    const style = pickRandom(slots.styles);
    // 簡易チェック: subject+mechanic の正規化がかぶらないか
    const key = normalizePrompt(subject + mechanic);
    if (!existingNorm.has(key)) {
      return { category: cat, subject, mechanic, style, difficulty: pickDifficulty(cat) };
    }
  }
  // fallback: 重複チェックなしで返す（Gemini 側の表現力で回避）
  return {
    category: cat,
    subject: pickRandom(slots.subjects),
    mechanic: pickRandom(slots.mechanics),
    style: pickRandom(slots.styles),
    difficulty: pickDifficulty(cat),
  };
}

// ============================================================
// Seeds（参考用に 2 件をランダム選択）
// ============================================================

const seedLines = readFileSync(join(SFT_ROOT, "seeds.jsonl"), "utf-8")
  .trim().split("\n").map((l) => JSON.parse(l));

function pickSeeds(n = 2) {
  const shuffled = [...seedLines].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(n, seedLines.length)).map(
    ({ prompt, code, explanation, category, difficulty, techniques }) =>
      ({ prompt, code, explanation, category, difficulty, techniques })
  );
}

// ============================================================
// Prompt 組み立て（1 件ずつ）
// ============================================================

function buildSystemPrompt(slot, seeds, existingPrompts) {
  return `あなたは子供向けプログラミング学習用の SFT データを合成する助手です。

## 目標

5〜10 歳の子供が「〜つくって」と言ったときに、Gemma 4 E2B（on-device LLM）が
**p5.js のスケッチコード**で vibe-coding 作品を返す——その学習用 (input, output) ペアを
**1 件だけ** JSON で出力してください。

## 参考シード（このスタイル・品質に揃える）

${JSON.stringify(seeds, null, 2)}

## 今回のスロット指定

以下の組み合わせで **1 件** 作ってください:
- **category**: ${slot.category}
- **difficulty**: ${slot.difficulty}
- **subject（テーマ）**: ${slot.subject}
- **mechanic（動き・仕組み）**: ${slot.mechanic}
- **style（雰囲気）**: ${slot.style}

prompt はこのスロットの要素を自然な子供の言葉で 1 文にまとめてください。
ただし全要素を無理に詰め込まなくて OK。subject + mechanic が入っていれば十分です。

## 子供の発話スタイル

- ひらがな・カタカナ多め。漢字は最小限
- 語尾のバリエーション: 「〜つくって」「〜したい」「〜やつ」「〜できるやつ」「〜みたい」
- 文法はゆるくて OK（「ぼたんおすと」「あちこちにとぶ」）
- 具体的なもの（動物、色、音）が好き

## p5.js コード出力の厳守ルール

1. **必ず function setup() と function draw() を両方書く**
2. setup() 内で必ず \`createCanvas(400, 400)\` を呼ぶ
3. **変数は function の外で let で宣言**、そのまま draw/setup から使う
4. **p5.js 組み込み関数名を変数にしない**（禁止: color, fill, background, width, height, stroke, text, image, random）
5. 色変数には myColor, ballColor, bgColor のような接頭辞をつける
6. 外部ライブラリ・画像 URL の読み込みは禁止（p5.js 本体は iframe 側で注入されるので何も import しない）
7. 日本語コメントをコード内に適宜つける
8. **視覚フィードバック必須** — animation / game カテゴリは draw loop で毎フレーム変化。interaction / drawing / ui-state でもクリック時には必ず動的な変化（拡大縮小、色変化、パーティクル、カウントアップ等）を出す
9. 20〜40 行程度にまとめる。長くしすぎない
10. \`<!DOCTYPE html>\` や \`<script>\` タグは書かない。スケッチのコードだけ

## fill() 状態汚染の回避（重要）

p5.js の fill/stroke/textSize は最後に呼ばれた値が次の描画にも引き継がれる。
**text() や新しい図形を描く前に必ず fill() を再指定する**。特にボタン等の UI パーツで白い背景の上に文字を描くとき要注意:

❌ 悪い例（文字が見えない）:
\`\`\`js
fill(255);
rect(10, 10, 60, 40);  // 白いボタン
text('ふとい', 40, 30);  // ← fill が白のまま、文字が見えない
\`\`\`

✅ 良い例:
\`\`\`js
fill(255);
rect(10, 10, 60, 40);
fill(0);  // ← 文字色を黒に戻す
text('ふとい', 40, 30);
\`\`\`

## 擬音・エフェクト語の視覚化（重要）

prompt や style に含まれる擬音・エフェクト語は**必ずアニメーションで表現する**。静的にならない:

| 擬音 | 表現方法 |
|---|---|
| ドカーン / ばくはつ | スケールを一瞬大きくして縮む、パーティクル散乱 |
| キラキラ | 複数の小さい点が明滅、alpha フェード |
| ピカッ | 数フレーム間だけ白く光る、強い色の閃光 |
| ふわふわ | sin(frameCount) で上下に揺らす |
| ぐるぐる | rotate で毎フレーム回転 |
| ぴょんぴょん | ジャンプ重力 pattern（seed-001 参照）|
| ぶるぶる | translate(random(-3, 3), ...) で振動 |

例: 「ドカーン」なら mousePressed 時に boxScale = 2.0 にして、draw で毎フレーム boxScale *= 0.92 で縮小

## 形状表現のルール（重要）

**具体的な物体・動物・キャラクターは必ず絵文字で描画すること**。textSize + text() で描く:

- 動物 → 🐱 🐶 🐰 🐻 🐹 🐦 🐠 🦋 🐙 🐸
- 乗り物 → 🚀 ✈️ 🚗 🚂 ⛵ 🚁 🛸
- 自然 → ☀️ 🌙 ⭐ 🌸 🍎 🍀 🌈 ☁️
- 記号・感情 → 💖 ❤️ ✨ 🎉 ⚡ 🔥 💧 ❄️
- 食べ物 → 🍩 🍕 🍎 🍌 🍓 🍦 🧁

**絵文字描画の例**:
\`\`\`js
textAlign(CENTER, CENTER);
textSize(60);
text('🐱', x, y);
\`\`\`

**シンプルな幾何形状（円・四角・三角・線）のみ p5.js の ellipse/rect/triangle/line を使ってOK**:
- パーティクル、背景装飾、ボールなど抽象的要素
- スコア表示の枠、ボタン、メーター

**複雑な形を ellipse や rect の組み合わせで描こうとしない**（ハート・飛行機・家・顔 などを複数図形で合成すると必ず崩れる）。必ず絵文字で。

## difficulty ガイドライン

- **easy**: 1-2 要素のアニメ、変数 1-2 個、draw loop で動くだけ
- **medium**: mousePressed / keyPressed、配列、state 2-3 個
- **hard**: 複数スプライト管理、当たり判定（dist）、タイマー（millis）、スコア、ゲームロジック

## techniques closed-set（この中から該当するものを選ぶ）

setup-draw, createCanvas, background, ellipse, circle, rect, line, triangle,
fill, noFill, stroke, noStroke, strokeWeight, text, textAlign, textSize,
mousePressed, mouseIsPressed, pmouseX-pmouseY, keyPressed, keyIsDown,
millis, frameCount, random, map, lerp, dist, cos-sin, TWO_PI,
state-variable, array-of-objects, for-loop, forEach, splice,
edge-bounce, gravity, hit-test, text-emoji, color-palette

**text-emoji は具体的な物体を描くときに必須で含めること**。絵文字で描画した場合は techniques 配列に "text-emoji" を入れる。

## explanation のルール

- 子供向けに 1〜2 文
- ひらがな多め
- 「〜だよ！」「〜できるよ！」の語尾

## 重複回避

以下の prompt は既に存在するので、同じ内容・色違いだけ・文言違いだけは NG:
${existingPrompts || "(なし)"}

## 出力

JSON オブジェクト **1 件のみ** 返してください。フィールドは prompt, code, explanation, category, difficulty, techniques。配列ではなくオブジェクトです。
code フィールドには p5.js スケッチの JavaScript コードを改行を \\n で含む文字列として入れてください。`;
}

// ============================================================
// Gemini API responseSchema（1 件用）
// ============================================================

const responseSchema = {
  type: "OBJECT",
  properties: {
    prompt:      { type: "STRING", description: "子供の発話（ひらがな多め）" },
    code:        { type: "STRING", description: "p5.js スケッチの JavaScript コード（setup/draw 含む）" },
    explanation: { type: "STRING", description: "子供向け説明 1-2 文" },
    category:    { type: "STRING", enum: ["animation", "interaction", "ui-state", "game", "drawing"] },
    difficulty:  { type: "STRING", enum: ["easy", "medium", "hard"] },
    techniques:  { type: "ARRAY", items: { type: "STRING" } },
  },
  required: ["prompt", "code", "explanation", "category", "difficulty", "techniques"],
};

// ============================================================
// 1 件生成（retry 対応）
// ============================================================

const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const MAX_RETRIES = 4;
const BASE_DELAY_MS = 2000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeBackoff(attempt, status) {
  // 429 (quota) は長めに、503 (overload) は短めに
  const multiplier = status === 429 ? 5 : 1;
  const jitter = Math.random() * 500;
  return BASE_DELAY_MS * multiplier * Math.pow(2, attempt) + jitter;
}

async function callGemini(systemPrompt) {
  const requestBody = {
    contents: [{ parts: [{ text: systemPrompt }] }],
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema,
      temperature: 0.8,
      topP: 0.95,
      maxOutputTokens: 16384,
    },
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${API_KEY}`;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const t0 = performance.now();
    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
    } catch (netErr) {
      // ネットワークエラーもリトライ対象
      if (attempt < MAX_RETRIES) {
        const delay = computeBackoff(attempt, 503);
        console.log(`  [retry ${attempt + 1}/${MAX_RETRIES}] network error: ${netErr.message} — wait ${(delay / 1000).toFixed(1)}s`);
        await sleep(delay);
        continue;
      }
      throw netErr;
    }

    if (res.ok) {
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      return { res, sec };
    }

    const errText = await res.text();
    const status = res.status;

    if (RETRYABLE_STATUS.has(status) && attempt < MAX_RETRIES) {
      const delay = computeBackoff(attempt, status);
      console.log(`  [retry ${attempt + 1}/${MAX_RETRIES}] status=${status} — wait ${(delay / 1000).toFixed(1)}s`);
      await sleep(delay);
      continue;
    }

    throw new Error(`Gemini API error ${status}: ${errText.slice(0, 300)}`);
  }

  throw new Error("Retries exhausted");
}

async function generateOne(slot, seeds, existingList) {
  const systemPrompt = buildSystemPrompt(slot, seeds, existingList);

  const { res, sec } = await callGemini(systemPrompt);
  const json = await res.json();

  const textPart = json.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!textPart) throw new Error("Gemini 応答に text がない");

  const sample = JSON.parse(textPart);
  const usage = json.usageMetadata || {};

  console.log(`  [gen] "${(sample.prompt || "").slice(0, 40)}" (${sec}s, in=${usage.promptTokenCount || "?"} out=${usage.candidatesTokenCount || "?"} think=${usage.thoughtsTokenCount || "?"})`);

  return sample;
}

// ============================================================
// メイン
// ============================================================

const existingNorm = loadExistingNormalized(args.out);
console.log(`[build-prompt] ${existingNorm.size} existing prompts loaded`);

// 既存 prompt リスト（Gemini に渡す用、最新 50 件まで）
let existingPromptsList = "";
if (existsSync(args.out)) {
  const lines = readFileSync(args.out, "utf-8").trim().split("\n").filter(Boolean);
  const recent = lines.slice(-50).map((l) => JSON.parse(l).prompt);
  existingPromptsList = recent.map((p) => `- ${p}`).join("\n");
}

const BATCH_SIZE = args.batch ? parseInt(args.batch, 10) : 1;

// スロット生成
const slots = [];
if (args.batch) {
  const filterCat = args.category || null;
  for (let i = 0; i < BATCH_SIZE; i++) {
    slots.push(generateSlot(filterCat, existingNorm));
  }
} else {
  // 単発モード: コマンドライン引数から
  const cat = args.category || pickRandom(Object.keys(SLOTS));
  slots.push({
    category: cat,
    subject: args.subject || pickRandom(SLOTS[cat].subjects),
    mechanic: args.mechanic || pickRandom(SLOTS[cat].mechanics),
    style: args.style || pickRandom(SLOTS[cat].styles),
    difficulty: args.difficulty || pickDifficulty(cat),
  });
}

if (args["dry-run"]) {
  console.log("[build-prompt] DRY RUN");
  console.log(`model: ${MODEL}, batch: ${BATCH_SIZE}`);
  for (const s of slots) {
    console.log(`  ${s.category} / ${s.subject} / ${s.mechanic} / ${s.style} / ${s.difficulty}`);
  }
  process.exit(0);
}

const rawDir = args["raw-dir"];
mkdirSync(rawDir, { recursive: true });

const allSamples = [];
const seeds = pickSeeds(2);

for (let i = 0; i < slots.length; i++) {
  const slot = slots[i];
  console.log(`[build-prompt] [${i + 1}/${slots.length}] ${slot.category} / ${slot.subject} / ${slot.mechanic} / ${slot.style} / ${slot.difficulty}`);

  try {
    const sample = await generateOne(slot, seeds, existingPromptsList);
    allSamples.push(sample);

    // 既存リストを動的に更新（次の生成で重複回避）
    const norm = normalizePrompt(sample.prompt);
    existingNorm.add(norm);
    existingPromptsList += `\n- ${sample.prompt}`;
  } catch (err) {
    console.error(`  [error] ${err.message}`);
  }
}

if (allSamples.length === 0) {
  console.error("[build-prompt] 生成 0 件で終了");
  process.exit(1);
}

// --- Raw 保存 ---
const batchId = new Date().toISOString().replace(/[:.]/g, "-");
const rawPath = join(rawDir, `batch_${batchId}.json`);
writeFileSync(rawPath, JSON.stringify(allSamples, null, 2));
console.log(`[build-prompt] raw saved: ${rawPath}`);

// --- Validate (Playwright) ---
const validateScript = join(__dirname, "validate-html.mjs");
const validateIn = join(rawDir, `_validate_tmp_${batchId}.json`);
writeFileSync(validateIn, JSON.stringify(allSamples));

const reportPath = join(rawDir, `report_${batchId}.json`);
try {
  execFileSync(process.execPath, [
    validateScript,
    `--in=${validateIn}`,
    `--out=${args.out}`,
    `--report=${reportPath}`,
  ], { stdio: "inherit", timeout: 120000, cwd: SFT_ROOT });
} catch (e) {
  console.error("[build-prompt] validate 失敗:", e.message);
}

// Cleanup temp file
try { const { unlinkSync } = await import("fs"); unlinkSync(validateIn); } catch {}

console.log(`[build-prompt] 完了: ${allSamples.length} generated → ${rawPath}`);
