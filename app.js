const MODEL_URL = "./models/gemma-4-E2B-it-web.task";
const MODEL_FILE = "gemma-4-E2B-it-web.task";

async function checkWebGPU() {
  if (!navigator.gpu) throw new Error("WebGPU 非対応ブラウザ");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("GPU adapter 取得失敗");
  const info = {
    maxBufferGB: (adapter.limits.maxBufferSize / 1e9).toFixed(2),
    f16: adapter.features.has("shader-f16"),
  };
  console.log("[vibeApp] WebGPU", info);
  return info;
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
    _nextId: 1,

    hints: [
      "ねこがはしるアニメをつくって",
      "ボタンで花火がでるやつ",
      "にじいろのボールがとぶやつ",
      "もぐらたたきしたい",
      "おえかきできるやつ",
    ],

    DARK_KEY: "vibe_dark_mode",

    async init() {
      try {
        this.darkMode = localStorage.getItem(this.DARK_KEY) === "1";
      } catch (e) {}

      try {
        const gpu = await checkWebGPU();
        if (Number(gpu.maxBufferGB) < 1.5) {
          this.statusText = "GPU メモリがたりないよ（Chrome 起動フラグを確認してね）";
          this.statusType = "error";
          return;
        }

        this.statusText = "AI のなかみを よみこんでるよ...";
        this.statusType = "downloading";

        try { await navigator.storage.persist(); } catch (e) {}

        const { FilesetResolver, LlmInference } = await import(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/genai_bundle.mjs"
        );
        const fileset = await FilesetResolver.forGenAiTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        const t0 = performance.now();
        const { stream, hit } = await loadModelWithCache(MODEL_URL, MODEL_FILE, ({ phase, bytes, total }) => {
          if (total) {
            this.loadPct = Math.round((bytes / total) * 100);
            this.statusText = hit
              ? "キャッシュから よみこんでるよ..."
              : "AI のなかみを ダウンロードちゅう " + this.loadPct + "%";
          }
        });

        this.statusText = "AI のあたまを くみたてちゅう...";
        this.statusType = "downloading";

        this.llm = await LlmInference.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: stream.getReader() },
          maxTokens: 2048,
          topK: 40,
          temperature: 0.7,
          randomSeed: 42,
        });

        const sec = ((performance.now() - t0) / 1000).toFixed(1);
        console.log("[vibeApp] LlmInference ready", { hit, sec });

        this.modelReady = true;
        this.statusText = "つくれるよ！";
        this.statusType = "ready";
        this.$nextTick(() => this.$refs.messageInput?.focus());
      } catch (err) {
        console.error("[vibeApp] init 失敗:", err);
        this.statusText = "じゅんびに しっぱいしちゃった...";
        this.statusType = "error";
      }
    },

    _systemPrompt() {
      // base model 用: 最小限の指示（SFT 後に Q1 詳細版に差し替え）
      return (
        "子供が「〜を作って」と言ったら、HTMLとCSSとJavaScriptで動くプログラムを1つ書いてください。\n" +
        "必ず1つの完全なHTMLファイルとして、```html で囲んで出力してください。\n" +
        "外部ライブラリは使わず、vanilla HTML/CSS/JS のみを使ってください。\n" +
        "CSS animationの@keyframesを積極的に使ってください。"
      );
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
      this.history = [];
      this.messages = [];
      this.previewCode = "";
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
      // 1) ```html ... ``` マーカーで囲まれたブロック
      const fenced = text.match(/```html\s*([\s\S]*?)```/);
      if (fenced) return fenced[1].trim();
      // 2) マーカーなしの raw HTML（<!DOCTYPE または <html で始まる）
      const raw = text.match(/(<!DOCTYPE[\s\S]*<\/html>)/i);
      if (raw) return raw[1].trim();
      return null;
    },

    openCodePreview(code) {
      this.previewCode = "";
      this.$nextTick(() => {
        this.previewCode = code;
      });
    },

    closeCodePreview() {
      this.previewCode = "";
    },

    async sendMessage() {
      const text = this.inputText.trim();
      if (!text || this.isGenerating || !this.modelReady) return;

      this.isGenerating = true;

      this.messages.push({
        id: this._nextId++,
        role: "user",
        text,
        isTyping: false,
      });

      this.inputText = "";
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

      try {
        const prompt = this._buildPrompt(text);
        console.log("[vibeApp] prompt length:", prompt.length);
        let fullText = "";

        if (typingIdx !== -1) this.messages[typingIdx].isTyping = false;

        await this.llm.generateResponse(prompt, (partial, done) => {
          fullText += partial;
          const cleaned = fullText.replace(/^>+\s*/, "");
          if (typingIdx !== -1) {
            this.messages[typingIdx].text = cleaned;
          }
          this.scrollToBottom();
        });

        // generateResponse 完了後（WASM コールバック外）でコード抽出・プレビュー
        const cleanedFull = fullText.replace(/^>+\s*/, "");

        const code = this.extractCodeBlock(cleanedFull);
        const explanation = cleanedFull.replace(/```[\s\S]*?```/g, "").trim() || "できたよ！";

        if (code && typingIdx !== -1) {
          this.messages[typingIdx].codeBlock = code;
          this.messages[typingIdx].text = explanation;
          this.openCodePreview(code);
        }
      } catch (err) {
        console.error("[vibeApp] generateResponse 失敗:", err);
        if (typingIdx !== -1) this.messages.splice(typingIdx, 1);
        this.messages.push({
          id: this._nextId++,
          role: "error",
          text: "😢",
          isTyping: false,
        });
      } finally {
        this.isGenerating = false;
        this.$nextTick(() => {
          this.scrollToBottom();
          this.$refs.messageInput?.focus();
        });
      }
    },
  }));
});
