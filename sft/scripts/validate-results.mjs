#!/usr/bin/env node
/**
 * Validate generated results against the p5.js execution sandbox (#163 / C-4).
 *
 * Notebook (Python) で生成した SFT 出力 JSON を入力に取り、
 * sft/eval-validate-runner.html を Playwright + Chrome (任意 GPU) で開いて
 * iframe sandbox で 5 秒 execute → exec_success_rate を集計する。
 *
 * Web 経路 (sft/scripts/evaluate.mjs) は LiteRT で base model のみ generate + validate するが、
 * Gemma 4 LoRA は LiteRT Web 公式未対応 (#129) のため、SFT 計測は本 script で
 * Notebook 経路 (Kaggle T4) と分離する。validate ロジックは eval-runner.html と同一実装。
 *
 * Input shape:
 *   { ts, model, system_prompt, n, results: [{ idx, prompt, category, difficulty, raw_text, gen_secs }] }
 * Output shape:
 *   { ..., exec_success_rate, mean_*, fail_reasons, results: [{ pass, reasons, ... }] }
 *
 * Usage:
 *   node sft/scripts/validate-results.mjs \
 *     --in=eval_phase4_sft.json \
 *     [--out=sft/logs/validate_<date>_<basename>.json] \
 *     [--port=8765] [--headed]
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, createReadStream } from "fs";
import { createServer } from "http";
import { extname, join, dirname, resolve, basename } from "path";
import { parseArgs } from "util";
import { chromium } from "playwright";

const { values: args } = parseArgs({
  options: {
    in:           { type: "string" },
    out:          { type: "string" },
    port:         { type: "string", default: "8765" },
    headed:       { type: "boolean", default: false },
    "user-data-dir": { type: "string", default: "sft/.chrome-profile" },
  },
});

if (!args.in) {
  console.error("Error: --in=<path> required");
  console.error("Usage: node sft/scripts/validate-results.mjs --in=<sft-output.json>");
  process.exit(1);
}

const ROOT = resolve(process.cwd());
const PORT = parseInt(args.port, 10);
const IN_PATH = resolve(args.in);
const IN_BASENAME = basename(IN_PATH);

if (!existsSync(IN_PATH)) {
  console.error(`Error: input not found: ${IN_PATH}`);
  process.exit(1);
}

console.log(`[validate] in: ${IN_PATH}`);

const MIME = {
  ".html": "text/html", ".js": "application/javascript", ".mjs": "application/javascript",
  ".css": "text/css", ".json": "application/json",
};
const server = createServer((req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${PORT}`);
    let p = decodeURIComponent(url.pathname);
    let full;
    if (p === `/_input/${IN_BASENAME}`) {
      full = IN_PATH;
    } else {
      if (p === "/") p = "/sft/eval-validate-runner.html";
      full = join(ROOT, p);
      if (!full.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
    }
    if (!existsSync(full) || !statSync(full).isFile()) { res.writeHead(404); res.end("not found"); return; }
    res.writeHead(200, {
      "Content-Type": MIME[extname(full)] || "application/octet-stream",
      "Content-Length": statSync(full).size,
      "Cache-Control": "no-store",
    });
    createReadStream(full).pipe(res);
  } catch (e) {
    res.writeHead(500); res.end(String(e));
  }
});
await new Promise((r) => server.listen(PORT, r));
console.log(`[validate] static server on http://localhost:${PORT}`);

const userDataDir = resolve(ROOT, args["user-data-dir"]);
mkdirSync(userDataDir, { recursive: true });
const context = await chromium.launchPersistentContext(userDataDir, {
  channel: "chrome",
  headless: !args.headed,
});

const page = context.pages()[0] || await context.newPage();
page.on("console", (msg) => {
  if (msg.type() === "error" || msg.text().startsWith("[validate]")) {
    console.log(`  [browser:${msg.type()}] ${msg.text()}`);
  }
});
page.on("pageerror", (err) => console.error(`  [browser:pageerror] ${err.message}`));

const url = `http://localhost:${PORT}/sft/eval-validate-runner.html`
  + `?data=${encodeURIComponent(`/_input/${IN_BASENAME}`)}`;
console.log(`[validate] navigate: ${url}`);
await page.goto(url);

await page.waitForFunction(() => !document.getElementById('run').disabled, null, { timeout: 30_000 });
console.log(`[validate] data loaded. starting validation...`);
await page.click('#run');

await page.waitForFunction(() => !!window.__VALIDATE_RESULTS__, null, { timeout: 30 * 60_000 });
const summary = await page.evaluate(() => window.__VALIDATE_RESULTS__);

const outPath = args.out
  || `sft/logs/validate_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}_${basename(IN_PATH, '.json')}.json`;
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(summary, null, 2));
console.log(`\n[validate] wrote ${outPath}`);
console.log(`  n=${summary.n}`);
console.log(`  exec_success_rate  = ${(summary.exec_success_rate * 100).toFixed(1)}%`);
console.log(`  mean_gen_secs      = ${summary.mean_gen_secs.toFixed(2)}`);
console.log(`  mean_code_length   = ${summary.mean_code_length.toFixed(0)}`);
console.log(`  has_animation_rate = ${(summary.has_animation_rate * 100).toFixed(1)}%`);
console.log(`  fail_reasons       =`, summary.fail_reasons);

if (!args.headed) {
  await context.close();
  server.close();
  process.exit(0);
} else {
  console.log(`[validate] --headed: leaving browser open. Ctrl+C to exit.`);
}
