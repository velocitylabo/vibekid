#!/usr/bin/env node
/**
 * SFT データ合成: Gemini API で 1 batch 生成 → validate → append
 *
 * Usage:
 *   node build-prompt.mjs --category=animation --difficulty=easy --n=5 \
 *     [--model=gemini-2.5-flash] [--out=data/sft.jsonl] [--raw-dir=data/raw] \
 *     [--dry-run]
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
    category:   { type: "string", default: "animation" },
    difficulty:  { type: "string", default: "easy" },
    n:           { type: "string", default: "5" },
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

const N = parseInt(args.n, 10);
const CATEGORY = args.category;
const DIFFICULTY = args.difficulty;
const MODEL = args.model;

// --- 1. Seeds からランダム 3 件 pick ---
const seedLines = readFileSync(join(SFT_ROOT, "seeds.jsonl"), "utf-8")
  .trim().split("\n").map((l) => JSON.parse(l));

function pickRandom(arr, n) {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(n, arr.length));
}

const pickedSeeds = pickRandom(seedLines, 3).map(({ id, prompt, html, explanation, category, difficulty, techniques }) =>
  ({ prompt, html, explanation, category, difficulty, techniques })
);

// --- 2. 既存 prompt 一覧（重複防止） ---
let existingPrompts = [];
if (existsSync(args.out)) {
  existingPrompts = readFileSync(args.out, "utf-8")
    .trim().split("\n").filter(Boolean)
    .map((l) => JSON.parse(l).prompt);
}
// seed prompts も追加
existingPrompts.push(...seedLines.map((s) => s.prompt));
const existingList = [...new Set(existingPrompts)].map((p) => `- ${p}`).join("\n");

// --- 3. System prompt 組み立て ---
const systemPrompt = `あなたは子供向けプログラミング学習用の SFT データを合成する助手です。

## 目標

5〜10 歳の子供が「〜つくって」と言ったときに、Gemma 4 E2B（on-device LLM）が
単一の HTML ファイルで vibe-coding 作品を返す——その学習用 (input, output) ペアを
JSON 配列で出力してください。

## 参考シード（このスタイル・品質に揃える）

${JSON.stringify(pickedSeeds, null, 2)}

## 今回のタスク

category="${CATEGORY}"、difficulty="${DIFFICULTY}" のサンプルを **${N} 件**、
JSON 配列で出してください。

## 子供の発話スタイル

- ひらがな・カタカナ多め。漢字は最小限
- 語尾のバリエーション: 「〜つくって」「〜したい」「〜やつ」「〜できるやつ」「〜みたい」
- 文法はゆるくて OK（「ぼたんおすと」「あちこちにとぶ」）
- 具体的なもの（動物、色、音）が好き

## HTML 出力の厳守ルール

1. \`<!DOCTYPE html>\` で始まる完全な単一ファイル
2. 外部ライブラリ・画像 URL・CDN は一切禁止。vanilla HTML/CSS/JS のみ
3. \`<script src="http...">\` や \`<img src="http...">\` は NG
4. 日本語コメントをコード内に付ける
5. CSS animation (\`@keyframes\`) を積極活用
6. 子供が見て「動いてる！」と思える視覚フィードバック必須

## difficulty ガイドライン

- **easy**: 1-2 要素 + 1 アニメ。JS 無しまたは最小
- **medium**: イベントハンドラ / DOM 動的生成 / state 1-2 個
- **hard**: 複数 interaction、setInterval/requestAnimationFrame、状態管理 3 個以上、配列操作

## techniques closed-set（この中から該当するものを選ぶ）

@keyframes, requestAnimationFrame, setInterval, setTimeout,
addEventListener, DOM-create, css-variables, classList-toggle,
transition, grid-layout, flex-layout, canvas-2d, Math-random,
state-variable, multiple-buttons, mouse-touch-unified, resize-handling,
forEach, array-management, viewport-units, emoji, linear-gradient,
animation-individual-params, color-picker-ui, clearRect, Web-Audio-API

## explanation のルール

- 子供向けに 1〜2 文
- ひらがな多め
- 「〜だよ！」「〜できるよ！」の語尾

## 重複回避

以下の prompt は既に存在するので、同じ内容・色違いだけ・文言違いだけは NG:
${existingList || "(なし)"}

## 出力

JSON 配列のみ返してください。他のテキストは不要です。`;

// --- 4. Gemini API responseSchema ---
const responseSchema = {
  type: "ARRAY",
  items: {
    type: "OBJECT",
    properties: {
      prompt:      { type: "STRING", description: "子供の発話（ひらがな多め）" },
      html:        { type: "STRING", description: "完全な単一 HTML ファイル" },
      explanation: { type: "STRING", description: "子供向け説明 1-2 文" },
      category:    { type: "STRING", enum: ["animation", "interaction", "ui-state", "game", "drawing"] },
      difficulty:  { type: "STRING", enum: ["easy", "medium", "hard"] },
      techniques:  { type: "ARRAY", items: { type: "STRING" } },
    },
    required: ["prompt", "html", "explanation", "category", "difficulty", "techniques"],
  },
};

// --- 5. Request body ---
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

if (args["dry-run"]) {
  console.log("[build-prompt] DRY RUN — prompt のみ表示");
  console.log(`model: ${MODEL}`);
  console.log(`category: ${CATEGORY}, difficulty: ${DIFFICULTY}, n: ${N}`);
  console.log(`picked seeds: ${pickedSeeds.map((s) => s.prompt).join(" / ")}`);
  console.log(`existing prompts: ${existingPrompts.length}`);
  console.log(`prompt length: ${systemPrompt.length} chars`);
  process.exit(0);
}

// --- 6. Call Gemini API ---
const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${API_KEY}`;

console.log(`[build-prompt] ${MODEL} / ${CATEGORY} / ${DIFFICULTY} / n=${N}`);
console.log(`[build-prompt] seeds: ${pickedSeeds.map((s) => s.prompt).join(" / ")}`);

const t0 = performance.now();
const res = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(requestBody),
});

if (!res.ok) {
  const errText = await res.text();
  console.error(`[build-prompt] Gemini API error ${res.status}:`, errText.slice(0, 500));
  process.exit(1);
}

const json = await res.json();
const sec = ((performance.now() - t0) / 1000).toFixed(1);

// --- 7. Parse response ---
const textPart = json.candidates?.[0]?.content?.parts?.[0]?.text;
if (!textPart) {
  console.error("[build-prompt] Gemini 応答に text がない:", JSON.stringify(json).slice(0, 500));
  process.exit(1);
}

let samples;
try {
  samples = JSON.parse(textPart);
} catch (e) {
  console.error("[build-prompt] JSON パース失敗:", e.message);
  console.error("[build-prompt] raw:", textPart.slice(0, 500));
  process.exit(1);
}

if (!Array.isArray(samples)) {
  console.error("[build-prompt] 配列ではない:", typeof samples);
  process.exit(1);
}

const usage = json.usageMetadata || {};
console.log(`[build-prompt] ${samples.length} samples generated in ${sec}s (tokens: in=${usage.promptTokenCount || "?"} out=${usage.candidatesTokenCount || "?"} think=${usage.thoughtsTokenCount || "?"})`);

// --- 8. Save raw ---
const rawDir = args["raw-dir"];
mkdirSync(rawDir, { recursive: true });
const batchId = new Date().toISOString().replace(/[:.]/g, "-");
const rawPath = join(rawDir, `batch_${batchId}.json`);
writeFileSync(rawPath, JSON.stringify(samples, null, 2));
console.log(`[build-prompt] raw saved: ${rawPath}`);

// --- 9. Validate (Playwright) ---
const validateScript = join(__dirname, "validate-html.mjs");
const validateIn = join(rawDir, `_validate_tmp_${batchId}.json`);
writeFileSync(validateIn, JSON.stringify(samples));

const reportPath = join(rawDir, `report_${batchId}.json`);
try {
  execFileSync(process.execPath, [
    validateScript,
    `--in=${validateIn}`,
    `--out=${args.out}`,
    `--report=${reportPath}`,
  ], { stdio: "inherit", timeout: 60000, cwd: SFT_ROOT });
} catch (e) {
  console.error("[build-prompt] validate 失敗:", e.message);
}

// Cleanup temp file
try { const { unlinkSync } = await import("fs"); unlinkSync(validateIn); } catch {}

console.log(`[build-prompt] 完了: ${rawPath} → validate → ${args.out}`);
