#!/usr/bin/env node
/**
 * SFT train/eval split generator
 *
 * sft.jsonl を Gemma 4 chat 形式に変換し、category 均等サンプリングで eval を切る。
 * システムプロンプトは user turn 1 に埋め込む A 案（project_mvp_day1_design.md Q2）。
 *
 * Usage:
 *   node prepare-train.mjs \
 *     --in=sft/data/sft.jsonl \
 *     --out-train=sft/data/train.jsonl \
 *     --out-eval=sft/data/eval.jsonl \
 *     [--eval-per-category=20] [--seed=42]
 *
 * Output schema (messages 形式, Unsloth apply_chat_template 前提):
 *   {
 *     "messages": [
 *       {"role": "user",      "content": "{SYSTEM_PROMPT}\n\n{prompt}"},
 *       {"role": "assistant", "content": "```js\n{code}\n```\n\n{explanation}"}
 *     ],
 *     "meta": { "category": ..., "difficulty": ..., "techniques": [...] }
 *   }
 */
import { readFileSync, writeFileSync } from "fs";
import { parseArgs } from "util";

const { values: args } = parseArgs({
  options: {
    in:                 { type: "string", default: "sft/data/sft.jsonl" },
    "out-train":        { type: "string", default: "sft/data/train.jsonl" },
    "out-eval":         { type: "string", default: "sft/data/eval.jsonl" },
    "eval-per-category":{ type: "string", default: "20" },
    seed:               { type: "string", default: "42" },
  },
});

// p5.js pivot 後の system prompt（MVP Day 1 design Q1 を p5.js 用に適応した短縮版）
// SFT 後の配信モデルで使う想定。one-shot example は base model 用のため SFT データには含めない
const SYSTEM_PROMPT = `あなたは子供向けプログラミングの先生AIです。

【絶対のルール】
1. 質問されたら、聞き返さず、すぐに 1 つだけ作って見せます。
2. 「赤と青どちらがいい?」のような選択肢は出しません。あなたが決めて作ります。
3. 出力は p5.js のスケッチコード 1 つだけ。\`\`\`js で囲みます。
4. 必ず setup() と draw() を書き、createCanvas(400, 400) を使います。
5. 変数は function の外で let で宣言します。
6. その後に 1〜2 文だけ、何を作ったかを優しい言葉で説明します。

【作るもののスタイル】
- 小学校低学年が見て楽しいもの
- 動きや色を多めに
- p5.js の組み込み関数（color, fill, background, width, height）を変数名にしない
- 外部ライブラリ・画像 URL・fetch・音源ファイルは使わない
- コードに日本語コメントを付ける

【絶対にしないこと】
- 同じ文を繰り返さない
- 「どんな〜にしますか」「〜と〜どちらがいいですか」と聞かない
- 解説を長く書かない（コード優先、説明は短く）`;

// --- Seeded RNG (mulberry32) ---
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, rng) {
  const a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// --- Load sft.jsonl ---
const lines = readFileSync(args.in, "utf-8").trim().split("\n").filter(Boolean);
const samples = lines.map((l) => JSON.parse(l));
console.log(`[prepare] loaded ${samples.length} samples from ${args.in}`);

// --- Group by category ---
const byCategory = new Map();
for (const s of samples) {
  const cat = s.category || "unknown";
  if (!byCategory.has(cat)) byCategory.set(cat, []);
  byCategory.get(cat).push(s);
}
console.log(`[prepare] categories:`);
for (const [cat, arr] of byCategory) {
  console.log(`  ${cat}: ${arr.length}`);
}

// --- Category 均等 eval split ---
const seed = parseInt(args.seed, 10);
const evalPerCategory = parseInt(args["eval-per-category"], 10);
const rng = mulberry32(seed);

const evalSamples = [];
const trainSamples = [];

for (const [cat, arr] of byCategory) {
  const shuffled = shuffle(arr, rng);
  const take = Math.min(evalPerCategory, shuffled.length);
  evalSamples.push(...shuffled.slice(0, take));
  trainSamples.push(...shuffled.slice(take));
}

// --- Transform to messages format ---
function toRecord(s) {
  const userContent = `${SYSTEM_PROMPT}\n\n${s.prompt}`;
  const assistantContent = `\`\`\`js\n${s.code}\n\`\`\`\n\n${s.explanation}`;
  return {
    messages: [
      { role: "user",      content: userContent },
      { role: "assistant", content: assistantContent },
    ],
    meta: {
      category:   s.category,
      difficulty: s.difficulty,
      techniques: s.techniques || [],
    },
  };
}

const trainShuffled = shuffle(trainSamples, rng);
const evalShuffled = shuffle(evalSamples, rng);

writeFileSync(args["out-train"], trainShuffled.map((s) => JSON.stringify(toRecord(s))).join("\n") + "\n");
writeFileSync(args["out-eval"],  evalShuffled.map((s) => JSON.stringify(toRecord(s))).join("\n") + "\n");

// --- Stats ---
const evalCatCounts = {};
const trainCatCounts = {};
for (const s of evalSamples)  evalCatCounts[s.category]  = (evalCatCounts[s.category]  || 0) + 1;
for (const s of trainSamples) trainCatCounts[s.category] = (trainCatCounts[s.category] || 0) + 1;

console.log(`\n[prepare] train: ${trainSamples.length}, eval: ${evalSamples.length}`);
console.log(`[prepare] train by category:`, trainCatCounts);
console.log(`[prepare] eval  by category:`, evalCatCounts);
console.log(`[prepare] wrote ${args["out-train"]} / ${args["out-eval"]}`);
