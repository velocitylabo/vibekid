document.addEventListener("alpine:init", () => {
  Alpine.data("vibeApp", () => ({
    messages: [],
    inputText: "",
    isGenerating: false,
    session: null,
    statusText: "じゅんびちゅう...",
    statusType: "",
    previewCode: "",
    darkMode: false,
    _nextId: 1,

    hints: [
      "ボールがはねるゲームをつくって",
      "カラフルな花火をつくって",
      "もぐらたたきをつくって",
      "おとがなるピアノをつくって",
    ],

    DARK_KEY: "vibe_dark_mode",

    async init() {
      try {
        this.darkMode = localStorage.getItem(this.DARK_KEY) === "1";
      } catch (e) { /* ignore */ }

      const api =
        typeof LanguageModel !== "undefined" ? LanguageModel :
        self.ai?.languageModel ? self.ai.languageModel :
        null;

      if (!api) {
        this.statusText = "Chrome Canary の Prompt API が つかえないよ";
        this.statusType = "error";
        return;
      }

      const languageOptions = {
        expectedInputs: [{ type: "text", languages: ["ja"] }],
        expectedOutputs: [{ type: "text", languages: ["ja"] }],
      };

      let availability;
      try {
        availability = await api.availability(languageOptions);
      } catch (err) {
        this.statusText = "エラー";
        this.statusType = "error";
        console.error("Availability check failed:", err);
        return;
      }

      if (availability === "unavailable") {
        this.statusText = "つかえないよ";
        this.statusType = "error";
        return;
      }

      try {
        this.session = await api.create({
          ...languageOptions,
          systemPrompt: this._systemPrompt(),
          monitor: (m) => {
            m.addEventListener("downloadprogress", (e) => {
              const pct = Math.round((e.loaded / e.total) * 100);
              this.statusText = "ダウンロードちゅう " + pct + "%";
              this.statusType = "downloading";
            });
          },
        });
        this.statusText = "つくれるよ！";
        this.statusType = "ready";
        this.$nextTick(() => this.$refs.messageInput?.focus());
      } catch (err) {
        this.statusText = "セッション さくせいに しっぱい";
        this.statusType = "error";
        console.error("Session creation failed:", err);
      }
    },

    _systemPrompt() {
      return "あなたは子供向けプログラミングの先生AIです。" +
        "子供が「〜を作って」「〜したい」と言ったら、HTMLとJavaScriptとCSSで動くプログラムを書いてください。" +
        "必ず1つの完全なHTMLファイルとして、```html で囲んで出力してください。" +
        "コードには日本語のコメントをつけてください。" +
        "小学校低学年にわかるように、やさしい言葉で説明してください。" +
        "コードの後に、1〜2文の短い説明を日本語でつけてください。" +
        "Canvas、CSS animation、Web Audio API などを活用してください。" +
        "外部ライブラリは使わず、vanilla HTML/CSS/JS のみを使ってください。";
    },

    toggleDarkMode() {
      this.darkMode = !this.darkMode;
      try {
        localStorage.setItem(this.DARK_KEY, this.darkMode ? "1" : "0");
      } catch (e) { /* ignore */ }
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
      const match = text.match(/```html\s*([\s\S]*?)```/);
      return match ? match[1].trim() : null;
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
      if (!text || this.isGenerating || !this.session) return;

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

      try {
        const stream = await this.session.promptStreaming(text);
        const typingIdx = this.messages.findIndex((m) => m.id === typingId);
        if (typingIdx !== -1) this.messages[typingIdx].isTyping = false;

        let fullText = "";
        for await (const chunk of stream) {
          fullText += chunk;
          if (typingIdx !== -1) {
            this.messages[typingIdx].text = fullText;
          }
          this.scrollToBottom();
        }

        const code = this.extractCodeBlock(fullText);
        if (code && typingIdx !== -1) {
          this.messages[typingIdx].codeBlock = code;
          const explanation = fullText.replace(/```[\s\S]*?```/g, "").trim();
          this.messages[typingIdx].text = explanation || "できたよ！";
          this.openCodePreview(code);
        }
      } catch (err) {
        console.error("Prompt error:", err);
        const typingIdx = this.messages.findIndex((m) => m.id === typingId);
        if (typingIdx !== -1) this.messages.splice(typingIdx, 1);
        this.messages.push({
          id: this._nextId++,
          role: "error",
          text: "ごめんね、エラーがおきちゃった。もういちど やってみてね。",
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
