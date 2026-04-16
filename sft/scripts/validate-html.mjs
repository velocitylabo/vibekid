#!/usr/bin/env node
/**
 * SFT データ合成用 HTML validator
 * Playwright headless Chromium でサンプルを実行し、品質チェックを行う
 *
 * Usage:
 *   node validate-html.mjs --in=batch.json --out=passed.jsonl [--report=report.json]
 *
 * Input: JSON array of {prompt, html, explanation, category, difficulty, techniques}
 * Output: 検証通過したサンプルだけ JSONL で出力
 */
import { chromium } from "playwright";
import { readFileSync, writeFileSync, appendFileSync } from "fs";
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

const TIMEOUT_MS = 4000;

const samples = JSON.parse(readFileSync(args.in, "utf-8"));
console.log(`[validate] ${samples.length} samples loaded from ${args.in}`);

const browser = await chromium.launch({ headless: true });
const results = [];

for (let i = 0; i < samples.length; i++) {
  const sample = samples[i];
  const label = `[${i + 1}/${samples.length}] "${(sample.prompt || "").slice(0, 30)}"`;
  const reasons = [];
  let pass = true;

  const html = sample.html || "";

  // --- Check 1: HTML minimum length ---
  if (html.length < 200) {
    reasons.push(`html_too_short (${html.length} chars)`);
    pass = false;
  }

  // --- Check 2: External resources ---
  if (/<(?:script|img|link|iframe)\s[^>]*src\s*=\s*["']https?:/i.test(html)) {
    reasons.push("external_resource");
    pass = false;
  }

  // --- Check 3: Animation check (for animation/game categories) ---
  const needsAnimation = ["animation", "game"].includes(sample.category);
  if (needsAnimation) {
    const hasAnimation =
      /@keyframes\s/.test(html) ||
      /requestAnimationFrame/.test(html) ||
      /setInterval/.test(html);
    if (!hasAnimation) {
      reasons.push("no_animation_detected");
      pass = false;
    }
  }

  // --- Static checks done, skip browser if already failed ---
  if (!pass) {
    console.log(`  ${label} FAIL (static) ${reasons.join(", ")}`);
    results.push({ ...sample, pass: false, reasons });
    continue;
  }

  // --- Check 4-6: Browser execution ---
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
    await page.goto(`file://${tmpPath}`, { timeout: TIMEOUT_MS, waitUntil: "load" });
    // Wait a bit for JS to execute
    await page.waitForTimeout(1500);

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

    // Check 6: Visible DOM
    const domInfo = await page.evaluate(() => {
      const body = document.body;
      if (!body) return { children: 0, width: 0, height: 0, textLen: 0 };
      const rect = body.getBoundingClientRect();
      return {
        children: body.children.length,
        width: rect.width,
        height: rect.height,
        textLen: body.innerText.length + body.innerHTML.length,
      };
    });

    if (domInfo.children === 0) {
      reasons.push("empty_body");
      pass = false;
    }
    if (domInfo.width === 0 || domInfo.height === 0) {
      reasons.push(`invisible (${domInfo.width}x${domInfo.height})`);
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

// Append passed samples to JSONL (without pass/reasons metadata)
for (const r of passed) {
  const { pass: _p, reasons: _r, ...clean } = r;
  appendFileSync(args.out, JSON.stringify(clean) + "\n");
}

console.log(`\n[validate] Done: ${passed.length} pass / ${failed.length} fail / ${results.length} total`);

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
    yieldRate: (passed.length / results.length * 100).toFixed(1) + "%",
    failReasons: reasonCounts,
  };
  writeFileSync(args.report, JSON.stringify(report, null, 2));
  console.log(`[validate] Report: ${args.report}`);
  console.log(report);
}
