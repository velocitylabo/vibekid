#!/usr/bin/env node
/**
 * SFT データ合成用 p5.js validator
 * Playwright headless Chromium でサンプルを実行し、品質チェックを行う
 *
 * Usage:
 *   node validate-html.mjs --in=batch.json --out=passed.jsonl [--report=report.json]
 *
 * Input: JSON array of {prompt, code, explanation, category, difficulty, techniques}
 * Output: 検証通過したサンプルだけ JSONL で出力
 */
import { chromium } from "playwright";
import { readFileSync, writeFileSync, appendFileSync, existsSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, unlink } from "fs/promises";
import { parseArgs } from "util";

const { values: args } = parseArgs({
  options: {
    in:     { type: "string" },
    out:    { type: "string", default: "passed.jsonl" },
    report: { type: "string" },
  },
});

if (!args.in) {
  console.error("Usage: node validate-html.mjs --in=batch.json [--out=passed.jsonl] [--report=report.json]");
  process.exit(1);
}

// --- Normalize prompt for dedup (NFKC + strip all whitespace) ---
function normalizePrompt(text) {
  return (text || "").normalize("NFKC").replace(/\s+/g, "");
}

// --- Load existing prompts for dedup ---
const existingNormalized = new Set();
if (existsSync(args.out)) {
  const lines = readFileSync(args.out, "utf-8").trim().split("\n").filter(Boolean);
  for (const line of lines) {
    existingNormalized.add(normalizePrompt(JSON.parse(line).prompt));
  }
}
console.log(`[validate] ${existingNormalized.size} existing prompts loaded for dedup`);

const TIMEOUT_MS = 6000;
const P5_CDN = "https://cdn.jsdelivr.net/npm/p5@1.11.3/lib/p5.min.js";

function wrapP5(code) {
  return `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta http-equiv="Permissions-Policy" content="accelerometer=(), gyroscope=(), magnetometer=()">
<script src="${P5_CDN}"><\/script>
<style>html,body{margin:0;padding:0;background:#fff}canvas{display:block}</style>
</head><body><script>
${code}
<\/script></body></html>`;
}

const samples = JSON.parse(readFileSync(args.in, "utf-8"));
console.log(`[validate] ${samples.length} samples loaded from ${args.in}`);

const browser = await chromium.launch({ headless: true });
const results = [];

for (let i = 0; i < samples.length; i++) {
  const sample = samples[i];
  const label = `[${i + 1}/${samples.length}] "${(sample.prompt || "").slice(0, 30)}"`;
  const reasons = [];
  let pass = true;

  const code = sample.code || "";

  // --- Check 1: Code minimum length ---
  if (code.length < 80) {
    reasons.push(`code_too_short (${code.length} chars)`);
    pass = false;
  }

  // --- Check 2: Required p5.js functions ---
  if (!/function\s+setup\s*\(\s*\)/.test(code)) {
    reasons.push("missing_setup");
    pass = false;
  }
  if (!/function\s+draw\s*\(\s*\)/.test(code)) {
    reasons.push("missing_draw");
    pass = false;
  }
  if (!/createCanvas\s*\(/.test(code)) {
    reasons.push("missing_createCanvas");
    pass = false;
  }

  // --- Check 3: No forbidden constructs ---
  if (/<script[\s>]|<\/script>/i.test(code)) {
    reasons.push("raw_html_in_code");
    pass = false;
  }
  if (/fetch\s*\(|XMLHttpRequest/i.test(code)) {
    reasons.push("network_call");
    pass = false;
  }

  // --- Static checks done, skip browser if already failed ---
  if (!pass) {
    console.log(`  ${label} FAIL (static) ${reasons.join(", ")}`);
    results.push({ ...sample, pass: false, reasons });
    continue;
  }

  // --- Browser execution ---
  const html = wrapP5(code);
  const tmpPath = join(tmpdir(), `sft_validate_${i}_${Date.now()}.html`);
  await writeFile(tmpPath, html, "utf-8");

  const context = await browser.newContext();
  const page = await context.newPage();

  const consoleErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  let pageError = null;
  page.on("pageerror", (err) => {
    pageError = err.message;
  });

  try {
    await page.goto(`file://${tmpPath}`, { timeout: TIMEOUT_MS, waitUntil: "networkidle" });
    // p5.js の setup/draw が動き出すまで待つ
    await page.waitForTimeout(2000);

    // Check 4: Page errors (JS exceptions)
    if (pageError) {
      reasons.push(`js_error: ${pageError.slice(0, 100)}`);
      pass = false;
    }

    // Check 5: Console errors
    if (consoleErrors.length > 0) {
      reasons.push(`console_errors (${consoleErrors.length}): ${consoleErrors[0].slice(0, 80)}`);
      pass = false;
    }

    // Check 6: Canvas が生成され描画されているか
    const canvasInfo = await page.evaluate(() => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return { hasCanvas: false };
      const rect = canvas.getBoundingClientRect();
      // Canvas 内容が空白でないか簡易チェック（全面同色じゃない）
      let nonBlank = false;
      try {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          // 全面 imageData を取得し、複数点をサンプリング
          const w = canvas.width;
          const h = canvas.height;
          const data = ctx.getImageData(0, 0, w, h).data;
          const first = (data[0] << 16) | (data[1] << 8) | data[2];
          const stride = 4 * 100;  // 100px 間隔でスキャン
          for (let i = stride; i < data.length; i += stride) {
            const px = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
            if (px !== first) { nonBlank = true; break; }
          }
        }
      } catch (_) { nonBlank = true; }  // WebGL canvas は getImageData 不可、無視
      return { hasCanvas: true, width: rect.width, height: rect.height, nonBlank };
    });

    // インタラクティブなスケッチ（mousePressed/keyPressed 等）なら blank を許容
    const isInteractive = /function\s+(mousePressed|mouseClicked|mouseReleased|keyPressed|keyReleased|touchStarted|touchEnded)\s*\(/.test(code);

    if (!canvasInfo.hasCanvas) {
      reasons.push("no_canvas");
      pass = false;
    } else if (canvasInfo.width === 0 || canvasInfo.height === 0) {
      reasons.push(`canvas_invisible (${canvasInfo.width}x${canvasInfo.height})`);
      pass = false;
    } else if (!canvasInfo.nonBlank && !isInteractive) {
      reasons.push("canvas_blank");
      pass = false;
    }
  } catch (err) {
    // Check 7: Timeout / infinite loop
    if (err.message.includes("Timeout") || err.message.includes("timeout")) {
      reasons.push("timeout_infinite_loop");
    } else {
      reasons.push(`browser_error: ${err.message.slice(0, 100)}`);
    }
    pass = false;
  } finally {
    await context.close();
    await unlink(tmpPath).catch(() => {});
  }

  const status = pass ? "PASS" : "FAIL";
  console.log(`  ${label} ${status}${reasons.length ? " " + reasons.join(", ") : ""}`);
  results.push({ ...sample, pass, reasons });
}

await browser.close();

// --- Output ---
const passed = results.filter((r) => r.pass);
const failed = results.filter((r) => !r.pass);

// Append passed samples to JSONL (without pass/reasons metadata), skipping duplicates
let appended = 0;
let dedupSkipped = 0;
for (const r of passed) {
  const { pass: _p, reasons: _r, ...clean } = r;
  const norm = normalizePrompt(clean.prompt);
  if (existingNormalized.has(norm)) {
    console.log(`  [dedup] skipped: "${(clean.prompt || "").slice(0, 40)}"`);
    dedupSkipped++;
    continue;
  }
  existingNormalized.add(norm);
  appendFileSync(args.out, JSON.stringify(clean) + "\n");
  appended++;
}

console.log(`\n[validate] Done: ${passed.length} pass / ${failed.length} fail / ${results.length} total / ${dedupSkipped} dedup-skipped / ${appended} appended`);

// Optional report
if (args.report) {
  const reasonCounts = {};
  for (const r of failed) {
    for (const reason of r.reasons) {
      const key = reason.split(":")[0].split("(")[0].trim();
      reasonCounts[key] = (reasonCounts[key] || 0) + 1;
    }
  }
  const report = {
    total: results.length,
    passed: passed.length,
    failed: failed.length,
    dedupSkipped,
    appended,
    yieldRate: (passed.length / results.length * 100).toFixed(1) + "%",
    appendRate: (appended / results.length * 100).toFixed(1) + "%",
    failReasons: reasonCounts,
  };
  writeFileSync(args.report, JSON.stringify(report, null, 2));
  console.log(`[validate] Report: ${args.report}`);
  console.log(report);
}
