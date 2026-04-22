const MODEL_URL = "./models/gemma-4-E2B-it-web.task";
const MODEL_FILE = "gemma-4-E2B-it-web.task";

// 旧版「AI おしゃべりひろば」時代の Service Worker / Cache Storage を一掃する。
// 以前アクセス済みのブラウザだけが対象。新規ユーザーには影響しない。
(async () => {
  if (!("serviceWorker" in navigator)) return;
  try {
    const regs = await navigator.serviceWorker.getRegistrations();
    if (regs.length === 0) return;
    await Promise.all(regs.map((r) => r.unregister()));
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    if (!sessionStorage.getItem("vibe_sw_purged")) {
      sessionStorage.setItem("vibe_sw_purged", "1");
      location.reload();
    }
  } catch (_) {}
})();

// 失敗モードを reason で分類し、init 側で user-visible メッセージに分岐する
async function checkWebGPU() {
  if (!navigator.gpu) return { ok: false, reason: "webgpu-unavailable" };
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", error: e };
  }
  if (!adapter) return { ok: false, reason: "adapter-unavailable" };
  const info = {
    maxBufferGB: (adapter.limits.maxBufferSize / 1e9).toFixed(2),
    f16: adapter.features.has("shader-f16"),
  };
  console.log("[vibeApp] WebGPU", info);
  if (Number(info.maxBufferGB) < 1.5) return { ok: false, reason: "low-buffer", info };
  if (!info.f16) return { ok: false, reason: "no-shader-f16", info };
  return { ok: true, info };
}

async function loadModelWithCache(url, fileName, onProgress) {
  const opfs = await navigator.storage.getDirectory();

  try {
    const fh = await opfs.getFileHandle(fileName);
    const sh = await opfs.getFileHandle(fileName + "_size");
    const file = await fh.getFile();
    const expected = parseInt(await (await sh.getFile()).text(), 10);
    if (Number.isFinite(expected) && file.size === expected) {
      onProgress?.({ phase: "cache-hit", bytes: file.size, total: expected });
      return { stream: file.stream(), hit: true };
    }
    await opfs.removeEntry(fileName).catch(() => {});
    await opfs.removeEntry(fileName + "_size").catch(() => {});
  } catch (_) {}

  const res = await fetch(url);
  if (!res.ok) throw new Error("モデル fetch 失敗: " + res.status);
  const total = parseInt(res.headers.get("Content-Length") || "0", 10);

  let bytesSeen = 0;
  const progressTap = new TransformStream({
    transform(chunk, ctrl) {
      bytesSeen += chunk.byteLength;
      onProgress?.({ phase: "fetch", bytes: bytesSeen, total });
      ctrl.enqueue(chunk);
    },
  });

  const [forConsumer, forCache] = res.body.pipeThrough(progressTap).tee();

  (async () => {
    try {
      const fh = await opfs.getFileHandle(fileName, { create: true });
      await forCache.pipeTo(await fh.createWritable());
      const sh = await opfs.getFileHandle(fileName + "_size", { create: true });
      const sw = await sh.createWritable();
      const finalSize = total || (await (await opfs.getFileHandle(fileName)).getFile()).size;
      await sw.write(String(finalSize));
      await sw.close();
      console.log("[vibeApp] OPFS 永続化完了");
    } catch (e) {
      console.warn("[vibeApp] OPFS 書き込み失敗", e);
    }
  })();

  return { stream: forConsumer, hit: false };
}

document.addEventListener("alpine:init", () => {
  Alpine.data("vibeApp", () => ({
    messages: [],
    inputText: "",
    isGenerating: false,
    llm: null,
    modelReady: false,
    history: [],
    statusText: "じゅんびちゅう...",
    statusType: "",
    loadPct: 0,
    previewCode: "",
    darkMode: false,
    voiceSupported: false,
    voiceReady: false,
    voiceInstalling: false,
    isRecording: false,
    voiceNotice: null,
    _nextId: 1,
    _heartbeatState: null,
    _lastUserText: "",
    _recognition: null,
    _voiceNoticeTimer: null,
    _voiceStartText: "",
    _p5ScriptTag: null,

    hints: [
      "ねこがぴょんぴょんはねるやつ",
      "ボタンおすといろがかわるやつ",
      "にじいろのボールがとぶやつ",
      "おえかきできるやつ",
      "ほしがキラキラひかるやつ",
    ],

    P5_CDN: "https://cdn.jsdelivr.net/npm/p5@1.11.3/lib/p5.min.js",

    DARK_KEY: "vibe_dark_mode",

    async init() {
      try {
        this.darkMode = localStorage.getItem(this.DARK_KEY) === "1";
      } catch (e) {}

      // p5.js を iframe に inline 展開できるよう先に取得（CDN 断でもプレビュー/validator を機能させる）
      this._p5ScriptTag = await this._buildP5ScriptTag();

      try {
        const gpu = await checkWebGPU();
        if (!gpu.ok) {
          const reasonMsg = {
            "webgpu-unavailable": "このブラウザは あたらしい AI に たいおうしてないみたい... さいしんの Chrome で ひらいてね",
            "adapter-error": "GPU の じゅんびで エラーが でたよ",
            "adapter-unavailable": "GPU が みつからなかったよ",
            "low-buffer": "GPU メモリが たりないよ（Chrome フラグ --enable-unsafe-webgpu を ゆうこうにしてね）",
            "no-shader-f16": "この かんきょうでは うごかないみたい（Windows の Chrome を おすすめするよ）",
          }[gpu.reason] || "GPU の じゅんびに しっぱいしたよ";
          this.statusText = reasonMsg;
          this.statusType = "error";
          console.warn("[vibeApp] WebGPU check failed:", gpu);
          return;
        }

        this.statusText = "ロボットくんを おこしてるよ...";
        this.statusType = "downloading";

        try { await navigator.storage.persist(); } catch (e) {}

        const { FilesetResolver, LlmInference } = await import(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/genai_bundle.mjs"
        );
        const fileset = await FilesetResolver.forGenAiTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        const llmOptions = {
          maxTokens: 2048,
          topK: 40,
          temperature: 0.7,
          randomSeed: 42,
        };

        const t0 = performance.now();

        // primary は OPFS + modelAssetBuffer、失敗時は modelAssetPath に fallback
        // ?forceFallback=1 で modelAssetPath 経路を強制再現（審査員環境検証用）
        const forceFallback = new URLSearchParams(location.search).get("forceFallback") === "1";
        let hit = false;
        try {
          if (forceFallback) throw new Error("forced fallback (query param)");
          const result = await loadModelWithCache(MODEL_URL, MODEL_FILE, ({ phase, bytes, total }) => {
            if (total) {
              this.loadPct = Math.round((bytes / total) * 100);
              this.statusText = hit
                ? "おぼえてた！ よみこんでるよ..."
                : "のうみそを ダウンロードちゅう... " + this.loadPct + "%";
            }
          });
          hit = result.hit;

          this.statusText = "のうみそを くみたてちゅう...";
          this.statusType = "downloading";

          this.llm = await LlmInference.createFromOptions(fileset, {
            ...llmOptions,
            baseOptions: { modelAssetBuffer: result.stream.getReader() },
          });
        } catch (bufferErr) {
          console.warn("[vibeApp] OPFS/modelAssetBuffer 経路 fallback:", bufferErr);
          this.statusText = "べつのほうほうで よみこみちゅう...";
          this.statusType = "downloading";
          this.loadPct = 0;

          this.llm = await LlmInference.createFromOptions(fileset, {
            ...llmOptions,
            baseOptions: { modelAssetPath: MODEL_URL },
          });
        }

        const sec = ((performance.now() - t0) / 1000).toFixed(1);
        console.log("[vibeApp] LlmInference ready", { hit, sec });

        this.modelReady = true;
        this.statusText = "じゅんびOK！";
        this.statusType = "ready";
        this._detectVoiceSupport();
        this.$nextTick(() => this.$refs.messageInput?.focus());
      } catch (err) {
        console.error("[vibeApp] init 失敗:", err);
        this.statusText = "じゅんびに しっぱいしちゃった...";
        this.statusType = "error";
      }
    },

    _systemPrompt() {
      // base model 用: one-shot 固定（SFT 後も one-shot で運用）
      // 4/17 spike で two-shot は 2B モデルを混乱させることを確認
      return `子供が「〜作って」と言ったら、p5.js のスケッチコードを1つだけ書いてください。

例（「ボールが跳ねる」の場合）:
\`\`\`js
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
\`\`\`

ルール:
- 変数は function の外で let で宣言する
- 関数は必ず setup() と draw() を書く
- createCanvas(400, 400) を使う
- background() を draw の最初に呼ぶ
- 変数名は ballX, bgColor, myScore のような分かりやすい名前を使う
- p5.js の組み込み関数名（color, fill, background, width, height）を変数名にしない
- 短く、動くコードだけ書く（20〜40行程度）
- \`\`\`js で囲んで出力する

同じスタイルで、指示されたものを作ってください:`;
    },

    _buildPrompt(userText) {
      // 履歴なし: 毎回 system + user input のみ送る
      // LlmInference の KV cache が呼び出し間で累積するため、
      // 履歴を含めると 2 回目以降で token 枯渇する
      // 会話的フォローアップは SFT 後に Session API で対応予定
      return (
        "<start_of_turn>user\n" +
        this._systemPrompt() + "\n\n" +
        userText +
        "<end_of_turn>\n" +
        "<start_of_turn>model\n"
      );
    },

    resetChat() {
      this.stopVoiceInput();
      this.history = [];
      this.messages = [];
      this.previewCode = "";
    },

    async _detectVoiceSupport() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR || typeof SR.available !== "function" || typeof SR.install !== "function") {
        this.voiceSupported = false;
        return;
      }
      try {
        const state = await SR.available({ langs: ["ja-JP"], processLocally: true });
        console.log("[vibeApp] voice availability:", state);
        if (state === "unavailable") {
          this.voiceSupported = false;
          return;
        }
        this.voiceSupported = true;
        this.voiceReady = state === "available";
      } catch (e) {
        console.warn("[vibeApp] voice availability check failed", e);
        this.voiceSupported = false;
      }
    },

    async _ensureVoiceInstalled() {
      if (this.voiceReady) return true;
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR?.install) return false;
      this.voiceInstalling = true;
      const t0 = performance.now();
      try {
        const ok = await SR.install({ langs: ["ja-JP"], processLocally: true });
        const sec = ((performance.now() - t0) / 1000).toFixed(1);
        console.log("[vibeApp] voice install", { ok, sec });
        if (!ok) {
          this._showVoiceNotice("⚠️", "こえの じゅんびに しっぱい");
          return false;
        }
        const state = await SR.available({ langs: ["ja-JP"], processLocally: true });
        this.voiceReady = state === "available";
        if (!this.voiceReady) {
          this._showVoiceNotice("⚠️", "こえの じゅんびに しっぱい");
        }
        return this.voiceReady;
      } catch (e) {
        console.error("[vibeApp] voice install failed", e);
        this._showVoiceNotice("⚠️", "こえの じゅんびに しっぱい");
        return false;
      } finally {
        this.voiceInstalling = false;
      }
    },

    async toggleVoiceInput() {
      if (this.voiceInstalling) return;
      if (this.isRecording) {
        this.stopVoiceInput();
        return;
      }
      if (this.isGenerating || !this.voiceSupported || !this.modelReady) return;
      if (!this.voiceReady) {
        const ok = await this._ensureVoiceInstalled();
        if (!ok) return;
      }
      this._startVoiceInput();
    },

    _startVoiceInput() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) return;
      let rec;
      try {
        rec = new SR();
      } catch (e) {
        console.error("[vibeApp] SpeechRecognition 生成失敗", e);
        this._handleSpeechError("start-failed");
        return;
      }
      rec.lang = "ja-JP";
      rec.continuous = false;
      rec.interimResults = true;
      try { rec.processLocally = true; } catch (_) {}

      this._voiceStartText = this.inputText;
      this.inputText = "";
      let finalText = "";

      rec.onresult = (e) => {
        let interim = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) finalText += t;
          else interim += t;
        }
        this.inputText = (finalText + interim).trim();
        this.$nextTick(() => this.resizeTextarea());
      };

      rec.onerror = (e) => {
        console.warn("[vibeApp] speech error:", e.error);
        this._handleSpeechError(e.error || "unknown");
      };

      rec.onend = () => {
        const wasRecording = this.isRecording;
        this.isRecording = false;
        this._recognition = null;
        if (!this.inputText.trim()) {
          if (this._voiceStartText) this.inputText = this._voiceStartText;
          if (wasRecording) this._showVoiceNotice("🤔", "きこえなかったよ");
        }
        this._voiceStartText = "";
        this.$nextTick(() => {
          this.resizeTextarea();
          this.$refs.messageInput?.focus();
        });
      };

      this._recognition = rec;
      this.isRecording = true;
      try {
        rec.start();
        console.log("[vibeApp] speech recognition started (ja-JP, on-device)");
      } catch (e) {
        console.error("[vibeApp] rec.start() failed", e);
        this.isRecording = false;
        this._recognition = null;
        this.inputText = this._voiceStartText;
        this._voiceStartText = "";
        this._handleSpeechError("start-failed");
      }
    },

    stopVoiceInput() {
      if (!this._recognition) return;
      try { this._recognition.stop(); } catch (_) {}
    },

    _handleSpeechError(code) {
      this.isRecording = false;
      this._recognition = null;
      if (this._voiceStartText) {
        if (!this.inputText.trim()) this.inputText = this._voiceStartText;
        this._voiceStartText = "";
      }
      if (code === "aborted") return;
      const map = {
        "not-allowed": ["🎤", "マイクをつかわせてね"],
        "service-not-allowed": ["🎤", "マイクをつかわせてね"],
        "no-speech": ["🤔", "きこえなかったよ"],
        "audio-capture": ["🎤", "マイクがみつからないよ"],
        "network": ["🌐", "インターネットがいるみたい"],
        "language-not-supported": ["😢", "にほんごは むり みたい"],
      };
      const [emoji, text] = map[code] || ["😢", "もういちど おしえて"];
      this._showVoiceNotice(emoji, text);
    },

    _showVoiceNotice(emoji, text) {
      if (this._voiceNoticeTimer) clearTimeout(this._voiceNoticeTimer);
      this.voiceNotice = { emoji, text };
      this._voiceNoticeTimer = setTimeout(() => {
        this.voiceNotice = null;
        this._voiceNoticeTimer = null;
      }, 2500);
    },

    toggleDarkMode() {
      this.darkMode = !this.darkMode;
      try {
        localStorage.setItem(this.DARK_KEY, this.darkMode ? "1" : "0");
      } catch (e) {}
    },

    handleKeydown(e) {
      if (e.key === "Enter" && !e.isComposing && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    },

    resizeTextarea() {
      const el = this.$refs.messageInput;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    },

    scrollToBottom() {
      const el = this.$refs.chatMessages;
      if (el) el.scrollTop = el.scrollHeight;
    },

    extractCodeBlock(text) {
      // p5.js モード: ```js または ```javascript ブロックを抽出
      const fenced = text.match(/```(?:js|javascript)\s*([\s\S]*?)```/);
      if (fenced) return fenced[1].trim();
      // fallback: マーカーなしで function setup() / draw() を含む素の JS
      const raw = text.match(/(function\s+setup\s*\(\)[\s\S]*?function\s+draw\s*\(\)[\s\S]*?)(?:\n\s*(?:```|$))/);
      if (raw) return raw[1].trim();
      // より緩い fallback: setup と draw の両方を含むテキスト全体
      if (/function\s+setup\s*\(\)/.test(text) && /function\s+draw\s*\(\)/.test(text)) {
        return text.trim();
      }
      return null;
    },

    async _buildP5ScriptTag() {
      const cdnTag = `<script src="${this.P5_CDN}"><\/script>`;
      try {
        const res = await fetch("./vendor/p5.min.js", { cache: "force-cache" });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const src = await res.text();
        // 将来 p5 に </script> 断片が混入しても HTML parser に閉じられないよう防御
        const safe = src.replace(/<\/script/gi, "<\\/script");
        console.log("[vibeApp] p5.js inline loaded", { bytes: src.length });
        return `<script>${safe}<\/script>`;
      } catch (e) {
        console.warn("[vibeApp] p5.js inline load failed, fallback to CDN", e);
        return cdnTag;
      }
    },

    wrapP5(code, tok) {
      // p5.js CDN をロードする iframe 用 HTML を組み立てる
      // canvas を iframe viewport に収める: flex center + max 100% で縦横比を保ったまま縮小
      // CSP: default-src 'none' で全遮断 → p5 CDN と inline のみ許可、connect-src で外部通信遮断
      // preHarness: 500ms 間隔で parent に heartbeat / error を post、hang 検知と runtime error 共有に使う
      const preHarness = `(() => {
        const tok = ${JSON.stringify(tok)};
        setInterval(() => { try { parent.postMessage({type:'vibe-heartbeat', tok}, '*'); } catch(_) {} }, 500);
        window.addEventListener('error', (e) => { try { parent.postMessage({type:'vibe-error', tok, msg:(e && e.message) || 'error'}, '*'); } catch(_) {} });
        window.addEventListener('unhandledrejection', (e) => { try { parent.postMessage({type:'vibe-error', tok, msg:String(e && e.reason)}, '*'); } catch(_) {} });
      })();`;
      return `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; frame-src 'none'; object-src 'none'; worker-src 'none'">
<meta http-equiv="Permissions-Policy" content="accelerometer=(), gyroscope=(), magnetometer=()">
<script>${preHarness}<\/script>
${this._p5ScriptTag}
<style>html,body{margin:0;padding:0;height:100%;background:#fff;overflow:hidden;display:flex;align-items:center;justify-content:center}canvas{display:block!important;width:auto!important;height:auto!important;max-width:100vw!important;max-height:100vh!important;object-fit:contain}</style>
</head><body><script>
${code}
<\/script></body></html>`;
    },

    openCodePreview(code) {
      // code は p5.js スニペット、wrapP5 で iframe 用 HTML に包む
      this._teardownHeartbeat();
      const tok = "p" + this._nextId++;
      this.previewCode = "";
      this.$nextTick(() => {
        this.previewCode = this.wrapP5(code, tok);
        this._startHeartbeatWatch(tok);
      });
    },

    _startHeartbeatWatch(tok) {
      // 初回 heartbeat まで 3s の grace（CDN fetch + 初期化を許容）
      // 初回受信後は 2s 無音で frozen 判定 → iframe をリセットしてエラー UI へ
      const startTime = Date.now();
      const state = { gotFirst: false, lastBeat: 0, frozen: false };
      const handler = (e) => {
        const d = e.data;
        if (!d || d.tok !== tok || state.frozen) return;
        if (d.type === "vibe-heartbeat") {
          state.gotFirst = true;
          state.lastBeat = Date.now();
        } else if (d.type === "vibe-error") {
          state.frozen = true;
          this._onPreviewError(d.msg);
        }
      };
      window.addEventListener("message", handler);
      const timerId = setInterval(() => {
        if (state.frozen) return;
        const now = Date.now();
        if (!state.gotFirst) {
          if (now - startTime > 3000) {
            state.frozen = true;
            this._onPreviewFrozen();
          }
        } else if (now - state.lastBeat > 2000) {
          state.frozen = true;
          this._onPreviewFrozen();
        }
      }, 500);
      this._heartbeatState = { tok, handler, timerId };
    },

    _teardownHeartbeat() {
      if (!this._heartbeatState) return;
      window.removeEventListener("message", this._heartbeatState.handler);
      clearInterval(this._heartbeatState.timerId);
      this._heartbeatState = null;
    },

    _onPreviewFrozen() {
      console.warn("[vibeApp] preview frozen (heartbeat timeout)");
      this.previewCode = "";
      this._teardownHeartbeat();
      if (this._lastUserText) this._pushErrorMessage("😵", this._lastUserText);
    },

    _onPreviewError(msg) {
      console.warn("[vibeApp] preview runtime error:", msg);
      this.previewCode = "";
      this._teardownHeartbeat();
      if (this._lastUserText) this._pushErrorMessage("🔧", this._lastUserText);
    },

    _pushErrorMessage(emoji, retryPrompt) {
      this.messages.push({
        id: this._nextId++,
        role: "error",
        emoji,
        retryPrompt,
        isTyping: false,
      });
      this.$nextTick(() => this.scrollToBottom());
    },

    retryPrompt(text) {
      if (this.isGenerating || !text) return;
      this.sendMessage(text);
    },

    // hidden sandboxed iframe で実行して、エラーなしに canvas + setup + draw が揃えば OK
    // error listener は user script より前に置いて同期エラーも拾う。
    // timeout は p5 初期化 + 実行時間を考慮して 5s（inline 経路は fetch ゼロ、CDN fallback 時のみ fetch 時間が乗る）
    _validateCode(code) {
      return new Promise((resolve) => {
        const tok = "v" + this._nextId++;
        const preHarness = `(() => {
          window.__VIBE_TOK = ${JSON.stringify(tok)};
          window.__VIBE_DONE = false;
          window.__VIBE_SEND = (p) => { if (window.__VIBE_DONE) return; window.__VIBE_DONE = true; parent.postMessage(Object.assign({type:'vibe-val', tok:window.__VIBE_TOK}, p), '*'); };
          window.addEventListener('error', (e) => window.__VIBE_SEND({ok:false, msg:(e && e.message) || 'error'}));
          window.addEventListener('unhandledrejection', (e) => window.__VIBE_SEND({ok:false, msg:String(e && e.reason)}));
        })();`;
        const postHarness = `(() => {
          setTimeout(() => {
            const hasCanvas = !!document.querySelector('canvas');
            const hasSetup = typeof setup === 'function';
            const hasDraw = typeof draw === 'function';
            if (hasCanvas && hasSetup && hasDraw) window.__VIBE_SEND({ok:true});
            else window.__VIBE_SEND({ok:false, msg:'canvas/setup/draw 未生成'});
          }, 1500);
        })();`;
        const srcdoc = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; frame-src 'none'; object-src 'none'; worker-src 'none'">
<script>${preHarness}<\/script>
${this._p5ScriptTag}
</head><body><script>${code}<\/script><script>${postHarness}<\/script></body></html>`;
        const iframe = document.createElement("iframe");
        iframe.setAttribute("sandbox", "allow-scripts");
        iframe.style.cssText = "position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;border:none";
        let settled = false;
        const cleanup = () => { if (settled) return; settled = true; try { iframe.remove(); } catch (_) {} window.removeEventListener("message", handler); };
        const handler = (e) => {
          const d = e.data;
          if (!d || d.type !== "vibe-val" || d.tok !== tok) return;
          cleanup();
          resolve({ ok: !!d.ok, msg: d.msg });
        };
        window.addEventListener("message", handler);
        setTimeout(() => { if (!settled) { cleanup(); resolve({ ok: false, msg: "timeout" }); } }, 5000);
        iframe.srcdoc = srcdoc;
        document.body.appendChild(iframe);
      });
    },

    closeCodePreview() {
      this._teardownHeartbeat();
      this.previewCode = "";
    },

    async sendMessage(retryText) {
      const text = (retryText || this.inputText).trim();
      if (!text || this.isGenerating || !this.modelReady) return;

      this.isGenerating = true;
      this._lastUserText = text;

      this.messages.push({
        id: this._nextId++,
        role: "user",
        text,
        isTyping: false,
      });

      if (!retryText) this.inputText = "";
      this.$nextTick(() => {
        this.resizeTextarea();
        this.scrollToBottom();
      });

      const typingId = this._nextId++;
      this.messages.push({
        id: typingId,
        role: "ai",
        text: "",
        isTyping: true,
        codeBlock: null,
      });
      this.$nextTick(() => this.scrollToBottom());

      const typingIdx = this.messages.findIndex((m) => m.id === typingId);

      let heartbeatId = null;
      try {
        const prompt = this._buildPrompt(text);
        console.log("[vibeApp] prompt length:", prompt.length);

        let fullText = "";
        let streamerFires = 0;
        let tFirstToken = 0;
        const tGenStart = performance.now();

        // 初回トークン到達まで 5s 毎に heartbeat (hang と TTFT 遅延の切り分け用)
        heartbeatId = setInterval(() => {
          if (tFirstToken === 0) {
            console.log("[vibeApp] generate heartbeat (pre-first-token)", {
              elapsedMs: Math.round(performance.now() - tGenStart),
            });
          }
        }, 5000);

        console.log("[vibeApp] generate start");
        await this.llm.generateResponse(prompt, (partial, done) => {
          streamerFires++;
          if (streamerFires === 1) {
            tFirstToken = performance.now();
            console.log("[vibeApp] first streamer fire (TTFT)", {
              ttftMs: Math.round(tFirstToken - tGenStart),
            });
            // 初回トークン到達時に typing indicator を外してテキスト表示へ
            if (typingIdx !== -1) this.messages[typingIdx].isTyping = false;
          }
          fullText += partial;
          const cleaned = fullText.replace(/^>+\s*/, "");
          if (typingIdx !== -1) {
            this.messages[typingIdx].text = cleaned;
          }
          this.scrollToBottom();
        });

        const tGenEnd = performance.now();
        const decodeMs = tFirstToken > 0 ? tGenEnd - tFirstToken : 0;
        const approxTokens = Math.round(fullText.length / 4); // 英語想定の概算 (#124 bench と同じ)
        console.log("[vibeApp] generate done", {
          totalMs: Math.round(tGenEnd - tGenStart),
          ttftMs: tFirstToken > 0 ? Math.round(tFirstToken - tGenStart) : "n/a",
          decodeMs: Math.round(decodeMs),
          streamerFires,
          outputChars: fullText.length,
          approxTokens,
          chunksPerSec: decodeMs > 0 ? (streamerFires / (decodeMs / 1000)).toFixed(1) : "n/a",
          decodeTokS: decodeMs > 0 ? (approxTokens / (decodeMs / 1000)).toFixed(1) : "n/a",
        });

        if (heartbeatId) { clearInterval(heartbeatId); heartbeatId = null; }
        // streamer が一度も発火しない異常ケースでも typing indicator を外す
        if (typingIdx !== -1 && this.messages[typingIdx].isTyping) {
          this.messages[typingIdx].isTyping = false;
        }

        // generateResponse 完了後（WASM コールバック外）でコード抽出・プレビュー
        const cleanedFull = fullText.replace(/^>+\s*/, "");

        const code = this.extractCodeBlock(cleanedFull);
        const explanation = cleanedFull.replace(/```[\s\S]*?```/g, "").trim() || "できたよ！";

        if (code && typingIdx !== -1) {
          const valid = await this._validateCode(code);
          if (valid.ok) {
            this.messages[typingIdx].codeBlock = code;
            this.messages[typingIdx].text = explanation;
            this.openCodePreview(code);
          } else {
            console.warn("[vibeApp] コード検証失敗:", valid.msg);
            if (typingIdx !== -1) this.messages.splice(typingIdx, 1);
            this._pushErrorMessage("🔧", text);
          }
        }
      } catch (err) {
        console.error("[vibeApp] generateResponse 失敗:", err);
        if (typingIdx !== -1) this.messages.splice(typingIdx, 1);
        this._pushErrorMessage("😢", text);
      } finally {
        if (heartbeatId) clearInterval(heartbeatId);
        this.isGenerating = false;
        this.$nextTick(() => {
          this.scrollToBottom();
          this.$refs.messageInput?.focus();
        });
      }
    },
  }));
});
