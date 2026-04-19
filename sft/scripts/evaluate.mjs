#!/usr/bin/env node
/**
 * Baseline / post-SFT eval runner
 *
 * sft/eval-runner.html を Playwright + Chrome stable (WebGPU) で開き、
 * eval.jsonl の prompt を 1 件ずつ Gemma 4 E2B web で生成 → iframe validate → 集計。
 *
 * Usage:
 *   node sft/scripts/evaluate.mjs \
 *     --eval=sft/data/eval.jsonl \
 *     --model=./models/gemma-4-E2B-it-web.task \
 *     --file=gemma-4-E2B-it-web.task \
 *     --out=sft/logs/eval_$(date +%Y%m%d)_base.json \
 *     [--port=8765] [--headed] [--limit=10] [--max-tokens=1024]
 *
 * 前提:
 *   - Linux + NVIDIA dGPU + Chrome stable（project_dev_machine.md の起動条件を Playwright 側で再現）
 *   - モデルは sft/ から相対で参照可能な場所に配置
 *   - Playwright の chromium が chrome channel で起動できる
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, createReadStream } from "fs";
import { createServer } from "http";
import { extname, join, dirname, resolve } from "path";
import { parseArgs } from "util";
import { chromium } from "playwright";

const { values: args } = parseArgs({
  options: {
    eval:          { type: "string", default: "sft/data/eval.jsonl" },
    model:         { type: "string", default: "./models/gemma-4-E2B-it-web.task" },
    file:          { type: "string", default: "gemma-4-E2B-it-web.task" },
    out:           { type: "string" },
    port:          { type: "string", default: "8765" },
    headed:        { type: "boolean", default: false },
    limit:         { type: "string" },
    "max-tokens":  { type: "string", default: "1024" },
    temperature:   { type: "string", default: "0.7" },
    "top-k":       { type: "string", default: "40" },
    "user-data-dir": { type: "string", default: "sft/.chrome-profile" },
  },
});

const ROOT = resolve(process.cwd());
const PORT = parseInt(args.port, 10);
const LIMIT = args.limit ? parseInt(args.limit, 10) : null;

// --- Load eval prompts ---
const lines = readFileSync(args.eval, "utf-8").trim().split("\n").filter(Boolean);
const allPrompts = lines.map((l) => {
  const r = JSON.parse(l);
  return {
    prompt:     r.messages?.[0]?.content?.split("\n\n").pop() || r.prompt,
    category:   r.meta?.category   ?? r.category,
    difficulty: r.meta?.difficulty ?? r.difficulty,
  };
});
const prompts = LIMIT ? allPrompts.slice(0, LIMIT) : allPrompts;
console.log(`[eval] loaded ${prompts.length} prompts from ${args.eval}${LIMIT ? ` (limit=${LIMIT})` : ""}`);

// --- Static file server (serve ROOT; allow model file too) ---
const MIME = {
  ".html": "text/html", ".js": "application/javascript", ".mjs": "application/javascript",
  ".css": "text/css", ".json": "application/json", ".task": "application/octet-stream",
  ".litertlm": "application/octet-stream", ".wasm": "application/wasm",
  ".jsonl": "application/x-ndjson", ".map": "application/json",
};
const server = createServer((req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${PORT}`);
    let p = decodeURIComponent(url.pathname);
    if (p === "/") p = "/sft/eval-runner.html";
    const full = join(ROOT, p);
    if (!full.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
    if (!existsSync(full) || !statSync(full).isFile()) { res.writeHead(404); res.end("not found"); return; }
    const size = statSync(full).size;
    res.writeHead(200, {
      "Content-Type": MIME[extname(full)] || "application/octet-stream",
      "Content-Length": size,
      "Cache-Control": "no-store",
    });
    createReadStream(full).pipe(res);
  } catch (e) {
    res.writeHead(500); res.end(String(e));
  }
});
await new Promise((r) => server.listen(PORT, r));
console.log(`[eval] static server on http://localhost:${PORT}`);

// --- Launch browser ---
// Chrome stable + Vulkan 経由 NVIDIA（project_dev_machine.md 参照）
// launchPersistentContext で OPFS を永続化、2 回目以降は cache HIT で即起動
const userDataDir = resolve(ROOT, args["user-data-dir"]);
mkdirSync(userDataDir, { recursive: true });
const context = await chromium.launchPersistentContext(userDataDir, {
  channel: "chrome",
  headless: !args.headed,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
    "--use-angle=vulkan",
    "--use-vulkan",
    "--ignore-gpu-blocklist",
    "--enable-dawn-features=allow_unsafe_apis",
  ],
});

// Inject eval prompts before any script runs
await context.addInitScript((data) => {
  window.__EVAL_PROMPTS__ = data;
}, prompts);

const page = context.pages()[0] || await context.newPage();
page.on("console", (msg) => {
  if (msg.type() === "error" || msg.text().startsWith("[eval]")) {
    console.log(`  [browser:${msg.type()}] ${msg.text()}`);
  }
});
page.on("pageerror", (err) => console.error(`  [browser:pageerror] ${err.message}`));

const url = `http://localhost:${PORT}/sft/eval-runner.html`
  + `?model=${encodeURIComponent("/" + args.model.replace(/^\.?\//, ""))}`
  + `&file=${encodeURIComponent(args.file)}`
  + `&max_tokens=${args["max-tokens"]}`
  + `&temp=${args.temperature}`
  + `&top_k=${args["top-k"]}`;
console.log(`[eval] navigate: ${url}`);
await page.goto(url);

// Click load, wait for ready
await page.click('#load');
console.log(`[eval] model loading (up to 60s on cold cache)...`);
await page.waitForFunction(() => !document.getElementById('run').disabled, null, { timeout: 600_000 });
console.log(`[eval] model ready. starting generation...`);

// Click run, wait for results
await page.click('#run');
console.log(`[eval] generating... (est. ${prompts.length * 20}s at ~20s/prompt)`);
await page.waitForFunction(() => !!window.__EVAL_RESULTS__, null, { timeout: 60 * 60_000 });

const summary = await page.evaluate(() => window.__EVAL_RESULTS__);

// --- Write output ---
const outPath = args.out
  || `sft/logs/eval_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}_base.json`;
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(summary, null, 2));
console.log(`\n[eval] wrote ${outPath}`);
console.log(`  n=${summary.n}`);
console.log(`  exec_success_rate  = ${(summary.exec_success_rate * 100).toFixed(1)}%`);
console.log(`  mean_gen_secs      = ${summary.mean_gen_secs.toFixed(2)}`);
console.log(`  mean_code_length   = ${summary.mean_code_length.toFixed(0)}`);
console.log(`  has_animation_rate = ${(summary.has_animation_rate * 100).toFixed(1)}%`);
console.log(`  mean_shape_calls   = ${summary.mean_shape_calls.toFixed(2)}`);
console.log(`  fail_reasons       =`, summary.fail_reasons);

if (!args.headed) {
  await context.close();
  server.close();
  process.exit(0);
} else {
  console.log(`[eval] --headed: leaving browser open for inspection. Ctrl+C to exit.`);
}
