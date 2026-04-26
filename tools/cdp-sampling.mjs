#!/usr/bin/env node
// 4 presets × N runs サンプリング runner via CDP
// memory feedback_sampling_protocol.md 準拠: 各 run hard reload + 単発 click
//
// 実行例:
//   node tools/cdp-sampling.mjs                    # default 5 runs/preset
//   RUNS_PER_PRESET=1 node tools/cdp-sampling.mjs  # smoke test
//   CDP_HOST=172.31.80.1:9222 APP_URL=http://localhost:8000 node tools/cdp-sampling.mjs

import { writeFile } from "node:fs/promises";

const CDP_HOST = process.env.CDP_HOST || "172.31.80.1:9222";
const APP_URL = process.env.APP_URL || "http://localhost:8000";
const RUNS_PER_PRESET = Number(process.env.RUNS_PER_PRESET || 5);
const POST_GENERATE_WAIT_MS = 5500; // preview heartbeat ok/frozen 判定が確定するまで

const PRESETS = [
  { text: "ねこがはしる", emoji: "🐱", category: "action" },
  { text: "ぼーるがはねる", emoji: "🏀", category: "action" },
  { text: "ボタンおすといろがかわる", emoji: "🔘", category: "interactive" },
  { text: "あめがふる", emoji: "☔", category: "visual" },
];

let cdpId = 0;
const pending = new Map();
let ws;

function send(method, params = {}) {
  const id = ++cdpId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, method });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function evalExpr(expr) {
  const r = await send("Runtime.evaluate", {
    expression: `(async () => { return (${expr}); })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (r.exceptionDetails) {
    const e = new Error("eval failed: " + (r.exceptionDetails.text || JSON.stringify(r.exceptionDetails)));
    e.cdpException = r.exceptionDetails;
    throw e;
  }
  return r.result.value;
}

async function poll(expr, timeoutMs = 60000, intervalMs = 250) {
  const t0 = Date.now();
  let lastErr;
  while (Date.now() - t0 < timeoutMs) {
    try {
      const v = await evalExpr(expr);
      if (v) return v;
    } catch (e) { lastErr = e; }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`poll timeout (${timeoutMs}ms): ${expr}` + (lastErr ? ` | last: ${lastErr.message}` : ""));
}

async function main() {
  // (1) discover or create target
  const targets = await fetch(`http://${CDP_HOST}/json`).then((r) => r.json());
  let target = targets.find((t) => t.type === "page" && t.url.startsWith(APP_URL));
  if (!target) {
    console.log("[cdp] no existing tab matching", APP_URL, "→ creating new");
    const res = await fetch(`http://${CDP_HOST}/json/new?${encodeURIComponent(APP_URL)}`, { method: "PUT" });
    target = await res.json();
  } else {
    console.log("[cdp] using existing tab:", target.url, target.id);
    // bring to front
    await fetch(`http://${CDP_HOST}/json/activate/${target.id}`).catch(() => {});
  }

  // (2) connect to page WS
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = (e) => rej(new Error("ws open error: " + (e.message || "unknown")));
  });
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(`CDP ${msg.error.code}: ${msg.error.message}`));
      else resolve(msg.result);
    }
  };
  ws.onclose = () => {
    for (const { reject } of pending.values()) reject(new Error("ws closed"));
    pending.clear();
  };

  await send("Page.enable");
  await send("Runtime.enable");

  // (3) wait Alpine + __vibeDiag exists, then clear
  await poll(`document.readyState === "complete"`, 30000);
  await poll(`!!window.__vibeDiag`, 60000);
  // wait for modelReady before starting (initial bootstrap may take long)
  await poll(`(() => { const el = document.querySelector("[x-data]"); return el && Alpine.$data(el).modelReady === true; })()`, 300000);
  await evalExpr(`window.__vibeDiag.clear()`);
  console.log("[cdp] init ok, diag cleared");

  // (4) sampling loop
  const runStart = Date.now();
  const total = PRESETS.length * RUNS_PER_PRESET;
  let runIdx = 0;
  for (let r = 0; r < RUNS_PER_PRESET; r++) {
    for (const preset of PRESETS) {
      runIdx++;
      const tag = `[${runIdx}/${total}] ${preset.text} run${r + 1}`;
      const tStart = Date.now();
      const preCount = await evalExpr(`window.__vibeDiag.size()`);

      console.log(tag, "reload");
      await send("Page.reload", { ignoreCache: true });

      await poll(`document.readyState === "complete"`, 30000);
      await poll(`!!window.__vibeDiag`, 60000);
      await poll(
        `(() => { const el = document.querySelector("[x-data]"); return el && Alpine.$data(el).modelReady === true && Alpine.$data(el).isGenerating === false; })()`,
        180000,
      );

      console.log(tag, "modelReady, click preset");
      await evalExpr(
        `Alpine.$data(document.querySelector("[x-data]")).selectPreset(${JSON.stringify(preset)})`,
      );

      // wait generation complete
      await poll(
        `Alpine.$data(document.querySelector("[x-data]")).isGenerating === false`,
        120000,
      );
      // diag entry pushed in finally block
      await poll(`window.__vibeDiag.size() > ${preCount}`, 5000);

      // wait preview annotation (heartbeat 2.5s ok / 3s frozen)
      await new Promise((r) => setTimeout(r, POST_GENERATE_WAIT_MS));

      // annotate run metadata for correlation
      await evalExpr(
        `Alpine.$data(document.querySelector("[x-data]"))._diagAnnotateLast(${JSON.stringify({
          _cdpMeta: { presetText: preset.text, runIndex: r + 1, sequenceIndex: runIdx },
        })})`,
      );

      const last = JSON.parse(await evalExpr(`JSON.stringify(window.__vibeDiag.list().slice(-1)[0])`));
      const elapsed = ((Date.now() - tStart) / 1000).toFixed(1);
      console.log(tag, `done in ${elapsed}s`, {
        ttftMs: last.ttftMs,
        totalMs: last.totalMs,
        outputChars: last.outputChars,
        validator: last.validator?.ok,
        preview: last.preview?.status,
        previewReason: last.preview?.reason || last.preview?.msg,
      });
    }
  }

  // (5) dump
  const json = await evalExpr(
    `JSON.stringify({ dumpedAt: new Date().toISOString(), version: "vibe-diag-1-cdp", entries: window.__vibeDiag.list() })`,
  );
  const ts = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outPath = `vibe-diag-cdp-${ts}.json`;
  await writeFile(outPath, json);
  const parsed = JSON.parse(json);
  const totalSec = ((Date.now() - runStart) / 1000).toFixed(0);
  console.log(`\n[cdp] saved ${outPath} entries=${parsed.entries.length} totalSec=${totalSec}`);

  ws.close();
  process.exit(0);
}

main().catch((e) => {
  console.error("[cdp] fatal:", e);
  process.exit(1);
});
