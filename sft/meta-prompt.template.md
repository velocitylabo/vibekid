# SFT データ合成 meta-prompt（Gemini 2.5 Pro/Flash 用）

## 使い方

`build-prompt.mjs` が以下のプレースホルダを埋めて、Gemini API に投げる:
- `{{SEEDS}}` — seeds.jsonl からランダム 3 件を JSON 配列で挿入
- `{{CATEGORY}}` — 今回の batch のカテゴリ（animation / interaction / ui-state / game / drawing）
- `{{DIFFICULTY}}` — easy / medium / hard
- `{{N}}` — 生成件数（通常 5）
- `{{EXISTING_PROMPTS}}` — 既存データの prompt 一覧（重複防止用、空文字可）

## Gemini API 設定

```json
{
  "model": "gemini-2.5-pro" or "gemini-2.5-flash",
  "generationConfig": {
    "responseMimeType": "application/json",
    "responseSchema": { ... },
    "temperature": 0.8,
    "topP": 0.95,
    "maxOutputTokens": 16384
  }
}
```

## responseSchema（Gemini JSON mode で強制）

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "prompt":      { "type": "string", "description": "子供の発話（ひらがな多め）" },
      "html":        { "type": "string", "description": "完全な単一 HTML ファイル" },
      "explanation": { "type": "string", "description": "子供向け説明 1-2 文" },
      "category":    { "type": "string", "enum": ["animation","interaction","ui-state","game","drawing"] },
      "difficulty":  { "type": "string", "enum": ["easy","medium","hard"] },
      "techniques":  { "type": "array", "items": { "type": "string" } }
    },
    "required": ["prompt","html","explanation","category","difficulty","techniques"]
  }
}
```

## prompt テンプレート本文

---

あなたは子供向けプログラミング学習用の SFT データを合成する助手です。

## 目標

5〜10 歳の子供が「〜つくって」と言ったときに、Gemma 4 E2B（on-device LLM）が
単一の HTML ファイルで vibe-coding 作品を返す——その学習用 (input, output) ペアを
JSON 配列で出力してください。

## 参考シード（このスタイル・品質に揃える）

```json
{{SEEDS}}
```

## 今回のタスク

category="{{CATEGORY}}"、difficulty="{{DIFFICULTY}}" のサンプルを **{{N}} 件**、
JSON 配列で出してください。

## 子供の発話スタイル

- ひらがな・カタカナ多め。漢字は最小限
- 語尾のバリエーション: 「〜つくって」「〜したい」「〜やつ」「〜できるやつ」「〜みたい」
- 文法はゆるくて OK（「ぼたんおすと」「あちこちにとぶ」）
- 具体的なもの（動物、色、音）が好き

## HTML 出力の厳守ルール

1. `<!DOCTYPE html>` で始まる完全な単一ファイル
2. 外部ライブラリ・画像 URL・CDN は一切禁止。vanilla HTML/CSS/JS のみ
3. `<script src="http...">` や `<img src="http...">` は NG
4. 日本語コメントをコード内に付ける
5. CSS animation (`@keyframes`) を積極活用
6. 子供が見て「動いてる！」と思える視覚フィードバック必須

## difficulty ガイドライン

- **easy**: 1-2 要素 + 1 アニメ。JS 無しまたは最小
- **medium**: イベントハンドラ / DOM 動的生成 / state 1-2 個
- **hard**: 複数 interaction、setInterval/requestAnimationFrame、状態管理 3 個以上、配列操作

## techniques closed-set（この中から該当するものを選ぶ）

```
@keyframes, requestAnimationFrame, setInterval, setTimeout,
addEventListener, DOM-create, css-variables, classList-toggle,
transition, grid-layout, flex-layout, canvas-2d, Math-random,
state-variable, multiple-buttons, mouse-touch-unified, resize-handling,
forEach, array-management, viewport-units, emoji, linear-gradient,
animation-individual-params, color-picker-ui, clearRect, Web-Audio-API
```

## explanation のルール

- 子供向けに 1〜2 文
- ひらがな多め
- 「〜だよ！」「〜できるよ！」の語尾

## 重複回避

以下の prompt は既に存在するので、同じ内容・色違いだけ・文言違いだけは NG:
{{EXISTING_PROMPTS}}

## 出力

上記 responseSchema に合致する JSON 配列を返してください。配列のみ、他のテキストは不要です。

---

## batch 実行計画

| batch 範囲 | category | difficulty | model | 件数 |
|---|---|---|---|---|
| 001-032 | animation (35%) | easy+medium | Gemini 2.5 Flash | 160 |
| 033-056 | interaction (25%) | easy+medium | Gemini 2.5 Flash | 120 |
| 057-064 | animation | medium+hard | Gemini 2.5 Flash | 40 |
| 065-080 | interaction | medium+hard | Gemini 2.5 Flash | 80 |
| 081-096 | ui-state (15%) | easy+medium | Gemini 2.5 Flash | 80 |
| 097-112 | drawing (10%) | easy+medium | Gemini 2.5 Flash | 80 |
| 113-128 | game (15%) | hard | **Gemini 2.5 Pro** | 80 |
| 129-144 | 全カテゴリ混合 | **hard** | **Gemini 2.5 Pro** | 80 |
| 145-160 | 穴埋め (yield 不足補填) | mixed | Gemini 2.5 Flash | 80 |

上記で 5 件 × 160 batch = 800 件（歩留まり 70% 前提で最終 560+ 件）。
不足分は batch 145-160 で穴埋め。
