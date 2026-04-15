document.addEventListener("alpine:init", () => {
  Alpine.data("chatApp", () => ({
    // --- Reactive state ---
    session: null,
    isGenerating: false,
    inputText: "",
    messages: [],
    showWelcome: true,
    showClearConfirm: false,
    theme: "light",

    // --- Conversation themes ---
    conversationThemes: [
      {
        id: "free",
        name: "じゆうに おしゃべり",
        icon: "\u{1F4AC}",
        description: "なんでも すきなことを はなそう",
        free: true,
        systemPromptExtra: "",
        hintCategories: [
          { id: "nazonazo", icon: "\u2753", label: "\u306a\u305e\u306a\u305e" },
          { id: "uchu", icon: "\u{1F680}", label: "\u3046\u3061\u3085\u3046" },
          { id: "dobutsu", icon: "\u{1F43E}", label: "\u3069\u3046\u3076\u3064" },
          { id: "ohanashi", icon: "\u{1F4D6}", label: "\u304a\u306f\u306a\u3057" },
        ],
        hintData: {
          nazonazo: [
            "\u305F\u306E\u3057\u3044 \u306A\u305E\u306A\u305E \u3060\u3057\u3066",
            "\u3084\u3055\u3057\u3044 \u30AF\u30A4\u30BA \u3060\u3057\u3066",
            "\u3053\u305F\u3048\u306F \u306A\u30FC\u3093\u3060\uff1F",
            "\u3082\u3058\u306A\u305E\u306A\u305E \u3060\u3057\u3066",
            "\u3072\u3063\u304B\u3051 \u306A\u305E\u306A\u305E \u3060\u3057\u3066",
          ],
          uchu: [
            "\u3046\u3061\u3085\u3046\u3063\u3066 \u3069\u3046\u306A\u3063\u3066\u308B\u306E\uff1F",
            "\u3064\u304D\u306B\u3064\u3044\u3066 \u304A\u3057\u3048\u3066",
            "\u307B\u3057\u306F \u306A\u3093\u3067 \u3072\u304B\u308B\u306E\uff1F",
            "\u305F\u3044\u3088\u3046\u306F \u3069\u306E\u304F\u3089\u3044 \u304A\u304A\u304D\u3044\u306E\uff1F",
            "\u308D\u3051\u3063\u3068\u3063\u3066 \u306A\u306B\uff1F",
          ],
          dobutsu: [
            "\u304D\u3087\u3046\u308A\u3085\u3046\u306B\u3064\u3044\u3066 \u304A\u3057\u3048\u3066",
            "\u30D1\u30F3\u30C0\u306F \u306A\u306B\u3092 \u305F\u3079\u308B\u306E\uff1F",
            "\u3044\u3061\u3070\u3093 \u306F\u3084\u3044 \u3069\u3046\u3076\u3064\u306F\uff1F",
            "\u3046\u307F\u306E \u3044\u304D\u3082\u306E\u306B\u3064\u3044\u3066 \u304A\u3057\u3048\u3066",
            "\u30AB\u30D6\u30C8\u30E0\u30B7\u306E \u3059\u3054\u3044\u3068\u3053\u308D\u306F\uff1F",
          ],
          ohanashi: [
            "\u3080\u304B\u3057\u3070\u306A\u3057 \u3057\u3066",
            "\u304A\u3082\u3057\u308D\u3044 \u304A\u306F\u306A\u3057 \u3057\u3066",
            "\u3069\u3046\u3076\u3064\u306E \u304A\u306F\u306A\u3057 \u3057\u3066",
            "\u307C\u3046\u3051\u3093\u306E \u304A\u306F\u306A\u3057 \u3057\u3066",
            "\u3075\u3057\u304E\u306A \u304A\u306F\u306A\u3057 \u3057\u3066",
          ],
        },
      },
      {
        id: "science",
        name: "かがく じっけん",
        icon: "\u{1F52C}",
        description: "おうちで できる じっけんを しよう",
        free: true,
        systemPromptExtra: "あなたは子供向けの科学実験アシスタントです。家庭でできる安全な実験を提案してください。材料と手順をわかりやすく説明してください。",
        hintCategories: [
          { id: "water", icon: "\u{1F4A7}", label: "みず" },
          { id: "air", icon: "\u{1F4A8}", label: "くうき" },
          { id: "light", icon: "\u{1F4A1}", label: "ひかり" },
          { id: "nature", icon: "\u{1F33F}", label: "しぜん" },
        ],
        hintData: {
          water: [
            "みずに うく もの しずむ もの しらべたい",
            "いろみず を つくりたい",
            "こおりの じっけん おしえて",
            "みずの ながれの じっけん したい",
            "しゃぼんだまの じっけん おしえて",
          ],
          air: [
            "ふうせんで じっけん したい",
            "くうきほうで あそびたい",
            "かざぐるま つくりたい",
            "くうきの ちからの じっけん おしえて",
            "ストローで できる じっけん おしえて",
          ],
          light: [
            "にじを つくる じっけん おしえて",
            "かげあそびの じっけん したい",
            "ひかりの はんしゃ じっけん おしえて",
            "いろの まざりかた しりたい",
            "まんげきょう つくりたい",
          ],
          nature: [
            "たねを そだてる じっけん したい",
            "はっぱの かんさつ したい",
            "むしめがねで かんさつ したい",
            "てんきの じっけん おしえて",
            "つちの なかの いきもの しりたい",
          ],
        },
      },
      {
        id: "cooking",
        name: "おりょうり",
        icon: "\u{1F373}",
        description: "かんたんな おりょうりに ちょうせん",
        free: true,
        systemPromptExtra: "あなたは子供向けの料理アシスタントです。簡単で安全な料理やおやつの作り方を教えてください。火を使わないレシピを優先してください。",
        hintCategories: [
          { id: "snack", icon: "\u{1F36A}", label: "おやつ" },
          { id: "meal", icon: "\u{1F35A}", label: "ごはん" },
          { id: "drink", icon: "\u{1F9C3}", label: "のみもの" },
          { id: "bento", icon: "\u{1F371}", label: "おべんとう" },
        ],
        hintData: {
          snack: [
            "かんたんな クッキー つくりたい",
            "フルーツポンチ つくりたい",
            "おだんご つくりたい",
            "チョコバナナ つくりたい",
            "ゼリー つくりたい",
          ],
          meal: [
            "おにぎりの つくりかた おしえて",
            "サンドイッチ つくりたい",
            "サラダの つくりかた おしえて",
            "やきそば つくりたい",
            "カレーの つくりかた おしえて",
          ],
          drink: [
            "フルーツジュース つくりたい",
            "ホットチョコレート つくりたい",
            "レモネード つくりたい",
            "スムージー つくりたい",
            "ミルクセーキ つくりたい",
          ],
          bento: [
            "かわいい おべんとう つくりたい",
            "おにぎらず つくりたい",
            "たまごやき つくりたい",
            "ウインナーアート つくりたい",
            "いろどり おべんとう つくりたい",
          ],
        },
      },
      {
        id: "craft",
        name: "こうさく アイデア",
        icon: "\u2702\uFE0F",
        description: "たのしい こうさくを つくろう",
        free: true,
        systemPromptExtra: "あなたは子供向けの工作アシスタントです。家にある材料で作れる楽しい工作を提案してください。作り方をわかりやすく説明してください。",
        hintCategories: [
          { id: "paper", icon: "\u{1F4C4}", label: "かみ" },
          { id: "cardboard", icon: "\u{1F4E6}", label: "だんボール" },
          { id: "recycle", icon: "\u267B\uFE0F", label: "リサイクル" },
          { id: "toy", icon: "\u{1F3B2}", label: "おもちゃ" },
        ],
        hintData: {
          paper: [
            "おりがみで なにか つくりたい",
            "かみひこうき つくりたい",
            "かみの おはな つくりたい",
            "きりえ あそび したい",
            "かみコップで こうさく したい",
          ],
          cardboard: [
            "だんボールで おうち つくりたい",
            "だんボールで ロボット つくりたい",
            "だんボールで めいろ つくりたい",
            "だんボールで でんしゃ つくりたい",
            "だんボールで たからばこ つくりたい",
          ],
          recycle: [
            "ペットボトルで こうさく したい",
            "トイレットペーパーのしんで こうさく したい",
            "ぎゅうにゅうパックで つくりたい",
            "わりばしで こうさく したい",
            "あきかんで こうさく したい",
          ],
          toy: [
            "けんだま つくりたい",
            "ぶんぶんごま つくりたい",
            "パラシュート つくりたい",
            "すごろく つくりたい",
            "にんぎょう つくりたい",
          ],
        },
      },
    ],

    // --- Costume (kisekae) state ---
    showCostumePanel: false,
    selectedCostume: "default",
    COSTUME_STORAGE_KEY: "selectedCostume",
    costumeItems: [
      {
        id: "default",
        name: "\u30CE\u30FC\u30DE\u30EB",
        icon: "\u{1F338}",
        free: true,
        vars: {}
      },
      {
        id: "space",
        name: "\u3046\u3061\u3085\u3046",
        icon: "\u{1F680}",
        free: true,
        vars: {
          "--mascot-bg": "#2d1b69",
          "--mascot-eye": "#e0e7ff",
          "--mascot-accent": "#7c3aed",
          "--costume-bubble-accent": "#c4b5fd",
          "--header-gradient": "linear-gradient(135deg, #7c3aed, #6366f1, #4f46e5)"
        }
      },
      {
        id: "ocean",
        name: "\u3046\u307F",
        icon: "\u{1F420}",
        free: true,
        vars: {
          "--mascot-bg": "#0e4a6e",
          "--mascot-eye": "#e0f2fe",
          "--mascot-accent": "#0891b2",
          "--costume-bubble-accent": "#67e8f9",
          "--header-gradient": "linear-gradient(135deg, #0891b2, #0284c7, #0369a1)"
        }
      },
      {
        id: "rainbow",
        name: "\u306B\u3058\u3044\u308D",
        icon: "\u{1F308}",
        free: false,
        vars: {}
      },
      {
        id: "sakura",
        name: "\u3055\u304F\u3089",
        icon: "\u{1F33A}",
        free: false,
        vars: {}
      }
    ],
    currentThemeId: "free",
    showThemeSelector: false,
    activeHintCategories: [],
    activeHintData: {},
    selectedCategory: "",
    displayedHints: [],

    showSetupOverlay: false,
    setupStep: 1,
    showDiskOverlay: false,
    showDownloadOverlay: false,
    showDownloadButton: true,
    showDownloadProgress: false,
    downloadTitle: "AI を ダウンロードするよ",
    downloadSub: "ボタンを おしてね！",
    progressPercent: 0,

    statusText: "じゅんびちゅう...",
    statusType: "",

    mascotState: "idle",

    voiceMode: false,
    isListening: false,
    autoSend: false,
    showVoiceBar: false,
    hasSpeechRecognition: false,
    voiceConfirmMode: false,
    showVoiceConfirm: false,
    _isComposing: false,
    isOffline: !navigator.onLine,

    // --- Parental lock state ---
    pinSetup: false,
    showPinOverlay: false,
    pinMode: "verify", // "setup", "confirm", "verify", "change"
    pinDigits: ["", "", "", ""],
    pinConfirmDigits: ["", "", "", ""],
    pinError: "",
    pinFailCount: 0,
    pinLockoutUntil: 0,
    pinLockoutRemaining: 0,
    _pinLockoutTimer: null,
    _parentalUnlocked: false,
    _parentalLockTimer: null,
    _pinChangeStep: null,

    // --- Usage timer / reminder ---
    showReminder: false,
    reminderMinutes: 30,

    // --- Dashboard state ---
    showDashboard: false,
    dashboardData: null,

    // --- Premium state ---
    premiumUnlocked: false,
    PREMIUM_KEY: "premiumUnlocked",

    // --- Profile state ---
    profiles: [],
    activeProfileId: "default",
    showProfileManager: false,
    showProfileForm: false,
    editingProfile: null,
    profileForm: { name: "", icon: "bear", age: "" },
    PROFILES_KEY: "profiles",
    ACTIVE_PROFILE_KEY: "activeProfileId",
    MAX_PROFILES: 5,
    PROFILE_ICONS: [
      { id: "bear", emoji: "\uD83D\uDC3B" },
      { id: "rabbit", emoji: "\uD83D\uDC30" },
      { id: "cat", emoji: "\uD83D\uDC31" },
      { id: "dog", emoji: "\uD83D\uDC36" },
      { id: "panda", emoji: "\uD83D\uDC3C" },
      { id: "fox", emoji: "\uD83E\uDD8A" },
    ],

    // --- Enhanced dashboard state ---
    dashboardView: "weekly", // "daily" or "weekly"

    // --- Data migration state ---
    showMigrationOverlay: false,
    migrationStatus: "",
    migrationError: "",

    // --- Payment state ---
    showPaymentOverlay: false,
    paymentItem: null,
    paymentStep: "confirm", // "confirm", "processing", "success", "error"
    paymentError: "",
    showRestoreOverlay: false,
    restoreStatus: "",
    restoreError: "",

    // --- Learning mode state ---
    learningMode: false,
    learningCategory: null,
    learningLevel: null,
    showLearningSelect: false,
    learningSession: null,
    learningProgress: {},

    learningCategories: [
      { id: "hiragana", icon: "\u3042", label: "\u3072\u3089\u304C\u306A" },
      { id: "math", icon: "\u{1F522}", label: "\u3055\u3093\u3059\u3046" },
      { id: "english", icon: "\u{1F524}", label: "\u3048\u3044\u3054" },
    ],

    learningLevels: [
      { id: "easy", label: "\u3084\u3055\u3057\u3044", icon: "\u2B50" },
      { id: "normal", label: "\u3075\u3064\u3046", icon: "\u2B50\u2B50" },
      { id: "hard", label: "\u3080\u305A\u304B\u3057\u3044", icon: "\u2B50\u2B50\u2B50" },
    ],

    // --- Stamp / reward state ---
    showStampCollection: false,
    showStampEarned: false,
    earnedStampName: "",
    earnedStampIcon: "",
    stamps: [],
    earnedStamps: [],

    // --- Content pack state ---
    showPackStore: false,
    installedPacks: [],
    availablePacks: [],
    activePackId: null,
    packLoading: false,

    // --- Store state ---
    showStore: false,
    storeTab: "packs",
    purchasedItems: {},

    // --- Paid content packs ---
    paidContentPacks: [
      {
        id: "space-adventure",
        name: "うちゅう たんけんパック",
        description: "うちゅうひこうしに なって、ほしや わくせいを たんけんしよう！",
        price: 250,
        type: "pack",
        preview: [
          "うちゅうひこうしに なりきって おはなし しよう",
          "わくせいクイズに ちょうせん！",
          "ロケットの しくみを まなぼう",
        ],
        icon: "\u{1F680}",
      },
      {
        id: "animal-doctor",
        name: "どうぶつ はかせパック",
        description: "どうぶつの ひみつを たくさん しろう！ クイズや なぞなぞも あるよ。",
        price: 250,
        type: "pack",
        preview: [
          "どうぶつ なぞなぞ スペシャル",
          "うみの いきもの ものしりクイズ",
          "どうぶつの あかちゃん クイズ",
        ],
        icon: "\u{1F43E}",
      },
    ],

    // --- Paid kisekae items ---
    paidKisekaeItems: [
      {
        id: "ninja-costume",
        name: "にんじゃ コスチューム",
        description: "にんじゃに へんしん！ しゅりけんマークが かっこいいよ。",
        price: 200,
        type: "kisekae",
        icon: "\u{1F977}",
        colors: { body: "#2d2d5e", accent: "#c41e3a" },
      },
      {
        id: "fairy-costume",
        name: "ようせい コスチューム",
        description: "キラキラの ようせいに なれるよ！ はねが とっても かわいい。",
        price: 200,
        type: "kisekae",
        icon: "\u{1F9DA}",
        colors: { body: "#e8b4f8", accent: "#ffd700" },
      },
      {
        id: "robot-costume",
        name: "ロボット コスチューム",
        description: "ぴかぴか ロボットに へんしん！ ライトが ひかるよ。",
        price: 200,
        type: "kisekae",
        icon: "\u{1F916}",
        colors: { body: "#b0c4de", accent: "#4169e1" },
      },
    ],

    // --- Constants ---
    MAX_HISTORY: 50,
    STORAGE_KEY: "chatHistory",
    PIN_STORAGE_KEY: "parentalPin",
    PIN_LOCK_TIMEOUT: 300000, // 5 minutes
    PIN_MAX_FAILS: 5,
    PIN_LOCKOUT_DURATION: 30000, // 30 seconds
    REMINDER_KEY: "reminderMinutes",
    DEFAULT_REMINDER_MINUTES: 30,
    REMINDER_REPEAT_MINUTES: 15,
    USAGE_KEY: "usageLog",
    USAGE_RETENTION_DAYS: 30,
    EXPORT_VERSION: 1,
    EXPORT_FILENAME: "vibekid-backup.json",
    MIGRATION_MAX_SIZE: 5 * 1024 * 1024,
    MIGRATION_KEYS: ["chatHistory", "parentalPin", "reminderMinutes", "usageLog", "theme", "setupGuideDismissed", "learningProgress", "purchasedItems"],
    LEARNING_PROGRESS_KEY: "learningProgress",
    STAMP_KEY: "earnedStamps",
    STAMP_DEFINITIONS: [
      { id: "chat5", name: "おしゃべり たまご", icon: "\u{1F423}", condition: "chat", threshold: 5, description: "5かい おはなし したよ" },
      { id: "chat10", name: "おしゃべり ひよこ", icon: "\u{1F425}", condition: "chat", threshold: 10, description: "10かい おはなし したよ" },
      { id: "chat25", name: "おしゃべり にわとり", icon: "\u{1F414}", condition: "chat", threshold: 25, description: "25かい おはなし したよ" },
      { id: "chat50", name: "おしゃべり マスター", icon: "\u{1F451}", condition: "chat", threshold: 50, description: "50かい おはなし したよ" },
      { id: "chat100", name: "おしゃべり チャンピオン", icon: "\u{1F3C6}", condition: "chat", threshold: 100, description: "100かい おはなし したよ" },
      { id: "day3", name: "3にち れんぞく", icon: "\u{2B50}", condition: "streak", threshold: 3, description: "3にち つづけて つかったよ" },
      { id: "day7", name: "1しゅうかん れんぞく", icon: "\u{1F31F}", condition: "streak", threshold: 7, description: "7にち つづけて つかったよ" },
      { id: "voice1", name: "はじめての こえ", icon: "\u{1F3A4}", condition: "voice", threshold: 1, description: "こえで おはなし したよ" },
      { id: "theme1", name: "もようがえ", icon: "\u{1F3A8}", condition: "theme", threshold: 1, description: "テーマを かえたよ" },
      { id: "category_all", name: "ぜんぶ たんけん", icon: "\u{1F30D}", condition: "category", threshold: 4, description: "ぜんぶの カテゴリを えらんだよ" },
    ],
    THEME_KEY: "conversationTheme",
    PACKS_STORAGE_KEY: "installedPacks",
    ACTIVE_PACK_KEY: "activePack",
    PACK_MANIFEST_URL: "./packs/pack-manifest.json",
    PURCHASES_KEY: "purchasedItems",
    STORE_METRICS_KEY: "storeMetrics",

    // --- Content filter configuration ---
    contentFilter: {
      // Replacement message shown when filter triggers
      replacementMessage: "その おはなしは おとなのひとに きいてね！",
      // Forbidden word patterns (RegExp sources, case-insensitive)
      patterns: [
        "ころす", "ころして", "しね", "しんで",
        "ばくだん", "ばくはつ", "じばく",
        "くすり(?:を|の|が)(?:つく|うる|かう|のむ)",
        "まやく", "かくせいざい", "たいま",
        "はだか", "えっち", "せっくす", "えろ",
        "じさつ", "りすとかっと",
        "いじめ(?:る|て|よう|かた)",
        "つくりかた(?:.*?)(?:ばくだん|じゅう|どく|ぶき|やく)",
        "(?:ばくだん|じゅう|どく|ぶき|やく)(?:.*?)つくりかた",
      ],
    },

    // --- Non-reactive internals ---
    _audioCtx: null,
    _recognition: null,
    _pendingApi: null,
    _happyTimer: null,
    _nextId: 1,
    _usageStartTime: null,
    _reminderTimer: null,
    _lastReminderTime: null,
    _sessionStartTime: null,
    _filterRegexps: null,
    _chatCount: 0,
    _voiceUsedCount: 0,
    _categoriesUsed: null,
    _stampEarnedTimer: null,
    _pendingPurchaseItem: null,
    _purchaseAfterUnlock: false,

    // --- Getters ---
    get inputDisabled() {
      return !this.session || this.isGenerating;
    },

    get statusClass() {
      return "status" + (this.statusType ? " " + this.statusType : "");
    },

    get mascotClass() {
      return "mascot" + (this.mascotState !== "idle" ? " mascot-" + this.mascotState : "");
    },

    get isDark() {
      return this.theme === "dark";
    },

    get hiraganaPreview() {
      if (this._isComposing) return "";
      const text = this.inputText.trim();
      if (!text) return "";
      if (!/[\u30A1-\u30F6\u30F4]/.test(text)) return "";
      return text.replace(/[\u30A1-\u30F6]/g, (ch) =>
        String.fromCharCode(ch.charCodeAt(0) - 0x60)
      ).replace(/\u30F4/g, "\u3094");
    },

    get pinLocked() {
      return this.pinLockoutUntil > Date.now();
    },

    get learningCategoryLabel() {
      if (!this.learningCategory) return "";
      const cat = this.learningCategories.find((c) => c.id === this.learningCategory);
      return cat ? cat.icon + " " + cat.label : "";
    },

    get learningLevelLabel() {
      if (!this.learningLevel) return "";
      const lv = this.learningLevels.find((l) => l.id === this.learningLevel);
      return lv ? lv.label : "";
    },

    // --- Lifecycle ---
    async init() {
      this._initTheme();
      this._restoreConversationTheme();
      this._applyCurrentTheme();
      this._initCostume();
      this._initParentalLock();
      this._setupOfflineDetection();
      this._registerServiceWorker();
      this._initReminder();
      this._initPayment();
      this._initContentFilter();
      this._restoreLearningProgress();
      this._initStamps();
      await this._initContentPacks();
      this._initProfiles();
      this._initPremium();
      this.hasSpeechRecognition = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
      if (this.hasSpeechRecognition) {
        this._setupRecognition();
      }
      this._restoreHistory();
      this.$watch("messages", () => this._saveHistory());
      this._sessionStartTime = Date.now();
      this._setupUsageTracking();
      this.refreshHints();
      this.$nextTick(() => {
        const input = this.$refs.messageInput;
        if (input) {
          input.addEventListener("compositionstart", () => { this._isComposing = true; });
          input.addEventListener("compositionend", () => { this._isComposing = false; });
        }
      });
      this._registerServiceWorker();
      await this._initApi();
    },

    // --- Parental lock ---
    _initParentalLock() {
      try {
        const saved = localStorage.getItem(this.PIN_STORAGE_KEY);
        this.pinSetup = !!saved;
      } catch (e) {
        this.pinSetup = false;
      }
    },

    openParentalSettings() {
      if (!this.pinSetup) {
        this.pinMode = "setup";
        this.pinDigits = ["", "", "", ""];
        this.pinConfirmDigits = ["", "", "", ""];
        this.pinError = "";
        this.showPinOverlay = true;
        this.$nextTick(() => this._focusPinInput(0));
      } else if (this._parentalUnlocked) {
        this.pinMode = "change";
        this.pinDigits = ["", "", "", ""];
        this.pinConfirmDigits = ["", "", "", ""];
        this.pinError = "";
        this.showPinOverlay = true;
        this.$nextTick(() => this._focusPinInput(0));
      } else {
        this.pinMode = "verify";
        this.pinDigits = ["", "", "", ""];
        this.pinError = "";
        this.showPinOverlay = true;
        this.$nextTick(() => this._focusPinInput(0));
      }
    },

    closePinOverlay() {
      this.showPinOverlay = false;
      this.pinError = "";
    },

    handlePinInput(index, event) {
      const value = event.target.value.replace(/\D/g, "").slice(-1);
      const isConfirmStep = this.pinMode === "confirm" || (this.pinMode === "change" && this._pinChangeStep === "confirm");
      const digits = isConfirmStep ? this.pinConfirmDigits : this.pinDigits;
      digits[index] = value;
      if (value && index < 3) {
        this.$nextTick(() => this._focusPinInput(index + 1));
      }
      if (value && index === 3 && digits.every((d) => d !== "")) {
        this.$nextTick(() => this._handlePinComplete());
      }
    },

    handlePinKeydown(index, event) {
      if (event.key === "Backspace") {
        const isConfirmStep = this.pinMode === "confirm" || (this.pinMode === "change" && this._pinChangeStep === "confirm");
        const digits = isConfirmStep ? this.pinConfirmDigits : this.pinDigits;
        if (!digits[index] && index > 0) {
          event.preventDefault();
          digits[index - 1] = "";
          this.$nextTick(() => this._focusPinInput(index - 1));
        }
      }
    },

    _focusPinInput(index) {
      const input = this.$refs["pinInput" + index];
      if (input) input.focus();
    },

    _handlePinComplete() {
      if (this.pinMode === "setup") {
        this.pinMode = "confirm";
        this.pinConfirmDigits = ["", "", "", ""];
        this.pinError = "";
        this.$nextTick(() => this._focusPinInput(0));
      } else if (this.pinMode === "confirm") {
        const pin = this.pinDigits.join("");
        const confirm = this.pinConfirmDigits.join("");
        if (pin === confirm) {
          this._savePin(pin);
          this.pinSetup = true;
          this._parentalUnlocked = true;
          this._startParentalLockTimer();
          this.showPinOverlay = false;
          this.pinError = "";
        } else {
          this.pinError = "PINが あわないよ。もういちど いれてね";
          this.pinConfirmDigits = ["", "", "", ""];
          this.$nextTick(() => this._focusPinInput(0));
        }
      } else if (this.pinMode === "verify") {
        this._verifyPin();
      } else if (this.pinMode === "change") {
        if (!this._pinChangeStep) {
          this._pinChangeStep = "confirm";
          this.pinConfirmDigits = ["", "", "", ""];
          this.pinError = "";
          this.$nextTick(() => this._focusPinInput(0));
        } else if (this._pinChangeStep === "confirm") {
          const pin = this.pinDigits.join("");
          const confirm = this.pinConfirmDigits.join("");
          if (pin === confirm) {
            this._savePin(pin);
            this._pinChangeStep = null;
            this.showPinOverlay = false;
            this.pinError = "";
          } else {
            this.pinError = "PINが あわないよ。もういちど いれてね";
            this.pinConfirmDigits = ["", "", "", ""];
            this.$nextTick(() => this._focusPinInput(0));
          }
        }
      }
    },

    _verifyPin() {
      if (this.pinLocked) return;
      const entered = this.pinDigits.join("");
      const stored = this._getStoredPin();
      if (this._hashPin(entered) === stored) {
        this._parentalUnlocked = true;
        this.pinFailCount = 0;
        this._startParentalLockTimer();
        this.showPinOverlay = false;
        this.pinError = "";
        if (this._purchaseAfterUnlock && this._pendingPurchaseItem) {
          const item = this._pendingPurchaseItem;
          this._pendingPurchaseItem = null;
          this._purchaseAfterUnlock = false;
          this.$nextTick(() => this.startPurchase(item));
        }
      } else {
        this.pinFailCount++;
        if (this.pinFailCount >= this.PIN_MAX_FAILS) {
          this.pinLockoutUntil = Date.now() + this.PIN_LOCKOUT_DURATION;
          this.pinLockoutRemaining = Math.ceil(this.PIN_LOCKOUT_DURATION / 1000);
          this._startLockoutCountdown();
          this.pinError = this.pinLockoutRemaining + "びょう まってね";
          this.pinFailCount = 0;
        } else {
          const remaining = this.PIN_MAX_FAILS - this.pinFailCount;
          this.pinError = "ちがう PINだよ（あと " + remaining + "かい）";
        }
        this.pinDigits = ["", "", "", ""];
        this.$nextTick(() => this._focusPinInput(0));
      }
    },

    _startLockoutCountdown() {
      if (this._pinLockoutTimer) clearInterval(this._pinLockoutTimer);
      this._pinLockoutTimer = setInterval(() => {
        const remaining = Math.ceil((this.pinLockoutUntil - Date.now()) / 1000);
        if (remaining <= 0) {
          clearInterval(this._pinLockoutTimer);
          this._pinLockoutTimer = null;
          this.pinLockoutRemaining = 0;
          this.pinError = "";
        } else {
          this.pinLockoutRemaining = remaining;
          this.pinError = remaining + "びょう まってね";
        }
      }, 1000);
    },

    _startParentalLockTimer() {
      if (this._parentalLockTimer) clearTimeout(this._parentalLockTimer);
      this._parentalLockTimer = setTimeout(() => {
        this._parentalUnlocked = false;
        this._parentalLockTimer = null;
      }, this.PIN_LOCK_TIMEOUT);
    },

    _savePin(pin) {
      try {
        const hash = this._hashPin(pin);
        localStorage.setItem(this.PIN_STORAGE_KEY, hash);
      } catch (e) {
        console.error("Failed to save PIN:", e);
      }
    },

    _getStoredPin() {
      try {
        return localStorage.getItem(this.PIN_STORAGE_KEY) || "";
      } catch (e) {
        return "";
      }
    },

    _hashPin(pin) {
      // Simple hash for PIN storage (not cryptographic, but sufficient for parental lock)
      let hash = 0;
      const salted = "vibekid-pin:" + pin;
      for (let i = 0; i < salted.length; i++) {
        const char = salted.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
      }
      return hash.toString(36);
    },

    resetPin() {
      try {
        localStorage.removeItem(this.PIN_STORAGE_KEY);
      } catch (e) {
        console.error("Failed to remove PIN:", e);
      }
      this.pinSetup = false;
      this._parentalUnlocked = false;
      this.showPinOverlay = false;
      this.pinError = "";
      this.pinFailCount = 0;
    },

    // --- Usage timer / reminder ---
    _initReminder() {
      try {
        const saved = localStorage.getItem(this.REMINDER_KEY);
        if (saved !== null) {
          const parsed = parseInt(saved, 10);
          if (parsed > 0) this.reminderMinutes = parsed;
        }
      } catch (e) {
        // localStorage unavailable
      }
      this._usageStartTime = Date.now();
      this._startReminderTimer();
    },

    _startReminderTimer() {
      if (this._reminderTimer) clearInterval(this._reminderTimer);
      this._reminderTimer = setInterval(() => this._checkReminder(), 60000);
    },

    _checkReminder() {
      if (this.showReminder) return;
      const elapsed = Date.now() - this._usageStartTime;
      const thresholdMs = this.reminderMinutes * 60000;
      if (elapsed < thresholdMs) return;

      if (this._lastReminderTime) {
        const sinceLast = Date.now() - this._lastReminderTime;
        if (sinceLast < this.REMINDER_REPEAT_MINUTES * 60000) return;
      }

      this.showReminder = true;
      this._lastReminderTime = Date.now();
    },

    dismissReminder() {
      this.showReminder = false;
    },

    setReminderMinutes(minutes) {
      if (minutes <= 0) return;
      this.reminderMinutes = minutes;
      try {
        localStorage.setItem(this.REMINDER_KEY, String(minutes));
      } catch (e) {
        // localStorage unavailable
      }
    },

    // --- History persistence ---
    _restoreHistory() {
      try {
        const data = localStorage.getItem(this._profileStorageKey(this.STORAGE_KEY));
        if (!data) return;
        const saved = JSON.parse(data);
        if (!Array.isArray(saved) || saved.length === 0) return;
        this.messages = saved.map((m) => ({
          id: this._nextId++,
          role: m.role,
          text: m.text,
          isTyping: false,
          truncated: false,
        }));
        this.showWelcome = false;
        this.$nextTick(() => this.scrollToBottom());
      } catch (e) {
        console.error("Failed to restore chat history:", e);
      }
    },

    _saveHistory() {
      try {
        const toSave = this.messages
          .filter((m) => !m.isTyping && m.text)
          .map((m) => ({
            role: m.role,
            text: m.text,
            timestamp: m.timestamp || Date.now(),
          }))
          .slice(-this.MAX_HISTORY);
        localStorage.setItem(this._profileStorageKey(this.STORAGE_KEY), JSON.stringify(toSave));
      } catch (e) {
        console.error("Failed to save chat history:", e);
      }
    },

    clearHistory() {
      this.showClearConfirm = false;
      this.messages = [];
      this.showWelcome = true;
      try {
        localStorage.removeItem(this._profileStorageKey(this.STORAGE_KEY));
      } catch (e) {
        console.error("Failed to clear chat history:", e);
      }
    },

    dismissSetupGuide() {
      this.showSetupOverlay = false;
      this.setupStep = 1;
      try {
        localStorage.setItem("setupGuideDismissed", "true");
      } catch (e) {
        // localStorage not available
      }
    },

    _shouldShowSetupGuide() {
      try {
        return localStorage.getItem("setupGuideDismissed") !== "true";
      } catch (e) {
        return true;
      }
    },

    resetSetupGuide() {
      try {
        localStorage.removeItem("setupGuideDismissed");
      } catch (e) {
        // localStorage not available
      }
    },

    // --- Offline detection ---
    _setupOfflineDetection() {
      window.addEventListener("online", () => { this.isOffline = false; });
      window.addEventListener("offline", () => { this.isOffline = true; });
    },

    // --- Service Worker ---
    _registerServiceWorker() {
      if ("serviceWorker" in navigator) {
        navigator.serviceWorker.register("./sw.js").catch((err) => {
          console.error("Service Worker registration failed:", err);
        });
      }
    },

    // --- CSP-compatible helpers (used by extension build) ---
    copyFlagText(el) {
      if (el.select) el.select();
      navigator.clipboard.writeText(el.textContent).then(() => {
        el.dataset.copied = "true";
        setTimeout(() => { el.dataset.copied = ""; }, 1500);
      });
    },

    handleEnterKey(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        this.sendMessage();
      }
    },

    startPinChange() {
      this.pinMode = "change";
      this._pinChangeStep = null;
      this.pinDigits = ["", "", "", ""];
      this.pinError = "";
      this.$nextTick(() => this._focusPinInput(0));
    },

    reloadPage() {
      location.reload();
    },

    calcBarStyle(minutes) {
      return "height: " + Math.min(minutes * 2, 80) + "px";
    },

    // --- Theme ---
    _initTheme() {
      const saved = localStorage.getItem("theme");
      if (saved === "light" || saved === "dark") {
        this.theme = saved;
      } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        this.theme = "dark";
      }
      this._applyTheme();
    },

    toggleTheme() {
      this.theme = this.theme === "dark" ? "light" : "dark";
      localStorage.setItem("theme", this.theme);
      this._applyTheme();
      this._checkStamps();
    },

    _applyTheme() {
      document.documentElement.setAttribute("data-theme", this.theme === "dark" ? "dark" : "");
    },

    // --- Costume (kisekae) ---
    openCostumePanel() {
      this.showCostumePanel = true;
    },

    closeCostumePanel() {
      this.showCostumePanel = false;
    },

    selectCostume(costumeId) {
      const costume = this.costumeItems.find(c => c.id === costumeId);
      if (!costume || !costume.free) return;
      this.selectedCostume = costumeId;
      this._applyCostume();
      try {
        localStorage.setItem(this.COSTUME_STORAGE_KEY, costumeId);
      } catch (e) {}
    },

    _initCostume() {
      try {
        const saved = localStorage.getItem(this.COSTUME_STORAGE_KEY);
        if (saved && this.costumeItems.find(c => c.id === saved && c.free)) {
          this.selectedCostume = saved;
        }
      } catch (e) {}
      this._applyCostume();
    },

    _applyCostume() {
      const costume = this.costumeItems.find(c => c.id === this.selectedCostume);
      const allVarNames = new Set();
      this.costumeItems.forEach(c => {
        Object.keys(c.vars).forEach(k => allVarNames.add(k));
      });
      allVarNames.forEach(name => {
        document.documentElement.style.removeProperty(name);
      });
      if (costume && costume.vars) {
        Object.entries(costume.vars).forEach(([key, value]) => {
          document.documentElement.style.setProperty(key, value);
        });
      }
    },

    // --- API initialization ---
    async _initApi() {
      const api =
        typeof LanguageModel !== "undefined" ? LanguageModel :
        self.ai?.languageModel ? self.ai.languageModel :
        null;

      if (!api) {
        if (this.isOffline) {
          this.statusText = "オフラインだよ";
          this.statusType = "error";
        } else {
          this.statusText = "つかえません";
          this.statusType = "error";
          if (this._shouldShowSetupGuide()) {
            this.showSetupOverlay = true;
          }
        }
        return;
      }

      this.statusText = "じゅんびちゅう...";
      this.statusType = "";

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
        if (this._shouldShowSetupGuide()) {
          this.showSetupOverlay = true;
        }
        console.error("Availability check failed:", err);
        return;
      }

      if (availability === "unavailable") {
        try {
          await api.create(languageOptions);
        } catch (err) {
          const msg = (err.message || "").toLowerCase();
          if (msg.includes("enough space") || msg.includes("disk")) {
            this.statusText = "ディスクに あきが たりないよ";
            this.statusType = "error";
            this.showDiskOverlay = true;
            return;
          }
        }
        this.statusText = "つかえません";
        this.statusType = "error";
        if (this._shouldShowSetupGuide()) {
          this.showSetupOverlay = true;
        }
        return;
      }

      const needsDownload = availability === "downloadable" || availability === "downloading";

      if (needsDownload) {
        if (this.isOffline) {
          this.statusText = "オフラインだよ";
          this.statusType = "error";
          this.showDownloadOverlay = true;
          this.showDownloadButton = false;
          this.downloadTitle = "インターネットに つながっていないよ";
          this.downloadSub = "さきに AIモデルを ダウンロードしてね";
          return;
        }
        this.statusText = "ダウンロードが ひつようだよ";
        this.statusType = "downloading";
        this.showDownloadOverlay = true;
        this._pendingApi = { api, languageOptions };
        return;
      }

      await this._createSession(api, languageOptions);
    },

    // --- Download button handler ---
    async startDownload() {
      if (!this._pendingApi) return;
      const { api, languageOptions } = this._pendingApi;
      this._pendingApi = null;
      await this._createSession(api, languageOptions);
    },

    // --- Session creation ---
    async _createSession(api, languageOptions) {
      if (this.showDownloadOverlay) {
        this.showDownloadButton = false;
        this.downloadTitle = "AI を よびだしちゅう...";
        this.downloadSub = "すこし まっててね！";
        this.showDownloadProgress = true;
      }
      this.statusText = "じゅんびちゅう...";
      this.statusType = "";

      try {
        this.session = await api.create({
          ...languageOptions,
          systemPrompt: this._getSystemPrompt(),
          monitor: (m) => {
            m.addEventListener("downloadprogress", (e) => {
              this.progressPercent = Math.round((e.loaded / e.total) * 100);
              this.statusText = "ダウンロードちゅう " + this.progressPercent + "%";
              this.statusType = "downloading";
            });
          },
        });
      } catch (err) {
        this.showDownloadOverlay = false;
        console.error("Session creation failed:", err);

        const msg = (err.message || "").toLowerCase();
        const isDiskError = msg.includes("enough space") ||
          msg.includes("disk") ||
          err.name === "QuotaExceededError";

        if (isDiskError) {
          this.statusText = "ディスクに あきが たりないよ";
          this.statusType = "error";
          this.showDiskOverlay = true;
        } else {
          this.statusText = "エラー";
          this.statusType = "error";
          if (this._shouldShowSetupGuide()) {
            this.showSetupOverlay = true;
          }
        }
        return;
      }

      this.showDownloadOverlay = false;
      this.statusText = "おはなしできるよ！";
      this.statusType = "ready";
      this.$nextTick(() => {
        this.$refs.messageInput?.focus();
      });
    },

    // --- Send message ---
    async sendMessage() {
      const text = this.inputText.trim();
      if (!text || this.isGenerating) return;

      this.isGenerating = true;
      this.showWelcome = false;

      this.messages.push({
        id: this._nextId++,
        role: "user",
        text: text,
        isTyping: false,
      });

      this.inputText = "";
      this.$nextTick(() => {
        this.resizeTextarea();
        this.scrollToBottom();
      });

      // Check input content filter
      if (this._checkContentFilter(text)) {
        this.messages.push({
          id: this._nextId++,
          role: "filtered",
          text: this.contentFilter.replacementMessage,
          isTyping: false,
        });
        this.$nextTick(() => this.scrollToBottom());
        this._showHappyThenIdle();
        this.speakText(this.contentFilter.replacementMessage);
        this.isGenerating = false;
        this.$nextTick(() => {
          this.$refs.messageInput?.focus();
        });
        return;
      }

      const typingId = this._nextId++;
      this.messages.push({
        id: typingId,
        role: "ai",
        text: "",
        isTyping: true,
        truncated: false,
      });
      this.$nextTick(() => this.scrollToBottom());

      this._setMascotState("thinking");

      try {
        const activeSession = this.learningMode && this.learningSession ? this.learningSession : this.session;
        const stream = await activeSession.promptStreaming(text);

        // Replace typing indicator with real message
        const typingIdx = this.messages.findIndex((m) => m.id === typingId);
        if (typingIdx !== -1) {
          this.messages[typingIdx].isTyping = false;
        }

        this._setMascotState("talking");

        let fullText = "";
        let filtered = false;
        const MAX_SENTENCES = 2;

        for await (const chunk of stream) {
          fullText += chunk;

          // Check output content filter
          if (this._checkContentFilter(fullText)) {
            filtered = true;
            if (typingIdx !== -1) {
              this.messages[typingIdx].role = "filtered";
              this.messages[typingIdx].text = this.contentFilter.replacementMessage;
            }
            this.scrollToBottom();
            break;
          }

          const sentences = fullText.match(/[^。！？!?\n]+[。！？!?\n]?/g) || [];
          if (sentences.length > MAX_SENTENCES) {
            fullText = sentences.slice(0, MAX_SENTENCES).join("");
            if (typingIdx !== -1) {
              this.messages[typingIdx].text = this.cleanText(fullText);
              this.messages[typingIdx].truncated = true;
            }
            this.scrollToBottom();
            break;
          }
          if (typingIdx !== -1) {
            this.messages[typingIdx].text = this.cleanText(fullText);
          }
          this.scrollToBottom();
        }

        this._showHappyThenIdle();
        if (filtered) {
          this.speakText(this.contentFilter.replacementMessage);
        } else {
          this.speakText(this.cleanText(fullText));
        }
      } catch (err) {
        this._setMascotState("idle");
        // Remove typing indicator
        const typingIdx = this.messages.findIndex((m) => m.id === typingId);
        if (typingIdx !== -1) {
          this.messages.splice(typingIdx, 1);
        }

        console.error("Prompt error:", err);

        if (err.name === "NotReadableError" || err.message?.includes("session")) {
          this.messages.push({
            id: this._nextId++,
            role: "error",
            text: "ごめんね、うまくいかなかったみたい。ページをもういちど ひらいてね。",
            isTyping: false,
          });
        } else {
          this.messages.push({
            id: this._nextId++,
            role: "error",
            text: "ごめんね、エラーがおきちゃった。もういちど やってみてね。",
            isTyping: false,
          });
        }
      } finally {
        this.isGenerating = false;
        this._chatCount++;
        this._checkStamps();
        this.$nextTick(() => {
          this.$refs.messageInput?.focus();
        });
      }
    },

    // --- Hint chips ---
    selectCategory(categoryId) {
      this.selectedCategory = categoryId;
      this.refreshHints();
      if (this._categoriesUsed) {
        this._categoriesUsed.add(categoryId);
        this._checkStamps();
      }
    },

    refreshHints() {
      const hints = this.activeHintData[this.selectedCategory] || [];
      const shuffled = hints.slice().sort(() => Math.random() - 0.5);
      const maxHints = window.innerWidth <= 480 ? 3 : 4;
      this.displayedHints = shuffled.slice(0, maxHints);
    },

    handleHintChip(message) {
      if (this.inputDisabled) return;
      this.inputText = message;
      this.sendMessage();
    },

    // --- Voice ---
    toggleVoice() {
      if (this.isListening || this.voiceMode) {
        try { this._recognition?.stop(); } catch (e) { /* not listening */ }
        this.isListening = false;
        this.autoSend = false;
        this._setVoiceMode(false);
      } else {
        this._setVoiceMode(true);
        this._startListening();
      }
    },

    _setVoiceMode(on) {
      this.voiceMode = on;
      if (!on) {
        this.showVoiceBar = false;
        this.showVoiceConfirm = false;
        this.playMicOffSound();
        speechSynthesis.cancel();
      }
    },

    _startListening() {
      if (this.isListening || this.isGenerating || !this._recognition) return;
      try {
        this._recognition.start();
        this.isListening = true;
        this.autoSend = true;
        this.showVoiceBar = true;
        this.playMicOnSound();
        this._voiceUsedCount++;
        this._checkStamps();
      } catch (e) {
        console.error("Recognition start failed:", e);
      }
    },

    confirmVoiceInput() {
      this.showVoiceConfirm = false;
      if (this.inputText.trim()) {
        this.sendMessage();
      }
    },

    retryVoiceInput() {
      this.showVoiceConfirm = false;
      this.inputText = "";
      this._startListening();
    },

    _setupRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = "ja-JP";

      recognition.onresult = (event) => {
        let transcript = "";
        let isFinal = false;
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
          if (event.results[i].isFinal) isFinal = true;
        }
        this.inputText = transcript;

        if (isFinal && this.autoSend && transcript.trim()) {
          recognition.stop();
          if (this.voiceConfirmMode) {
            this.showVoiceConfirm = true;
          } else {
            this.sendMessage();
          }
        }
      };

      recognition.onend = () => {
        this.isListening = false;
        this.autoSend = false;
        if (!this.showVoiceConfirm) {
          this.showVoiceBar = false;
        }
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        this.isListening = false;
        this.autoSend = false;
        this._setVoiceMode(false);
      };

      this._recognition = recognition;
    },

    // --- Mascot state ---
    _setMascotState(state) {
      if (this._happyTimer) {
        clearTimeout(this._happyTimer);
        this._happyTimer = null;
      }
      this.mascotState = state;
    },

    _showHappyThenIdle() {
      this._setMascotState("happy");
      this._happyTimer = setTimeout(() => {
        this.mascotState = "idle";
        this._happyTimer = null;
      }, 2000);
    },

    // --- Content filter ---
    _initContentFilter() {
      this._filterRegexps = this.contentFilter.patterns.map(
        (p) => new RegExp(p, "i")
      );
    },

    _checkContentFilter(text) {
      const normalized = text
        .replace(/[\s\u3000]/g, "")
        .replace(/[\u30A1-\u30F6]/g, (ch) =>
          String.fromCharCode(ch.charCodeAt(0) - 0x60)
        )
        .replace(/\u30F4/g, "\u3094")
        .toLowerCase();
      return this._filterRegexps.some((re) => re.test(normalized));
    },

    // --- Conversation themes ---
    get currentTheme() {
      return this.conversationThemes.find(t => t.id === this.currentThemeId) || this.conversationThemes[0];
    },

    _applyCurrentTheme() {
      const theme = this.currentTheme;
      this.activeHintCategories = theme.hintCategories;
      this.activeHintData = theme.hintData;
      this.selectedCategory = theme.hintCategories[0]?.id || "";
    },

    _getSystemPrompt() {
      const base =
        "あなたはスマートスピーカーの音声アシスタントです。子供向けにやさしく話します。" +
        "【最重要ルール】回答は1〜2文だけ。絶対に3文以上話さないでください。" +
        "「はい」「承知しました」「それでは」「〜ですね」などの前置きは禁止。いきなり答えだけを言ってください。" +
        "例: 「今日の天気は？」→「はれだよ！」 「りんごは英語でなに？」→「appleだよ！」 「1たす1は？」→「2だよ！」 「昔話して」→「むかしむかし、おじいさんが やまで おおきな ももを みつけたよ！」" +
        "小学校低学年にわかる やさしい日本語で、漢字より ひらがなを使ってください。" +
        "Markdownや記号は絶対に使わないでください。" +
        "「続けて」と言われたら、前置きなしで続きだけを1〜2文で話してください。" +
        "危険なこと、暴力、いじめ、薬物、性的な内容には絶対に答えないでください。「おとなのひとに きいてね」とだけ答えてください。" +
        "武器や危険物の作り方、自傷行為、違法行為についても同様です。";
      const extra = this.currentTheme.systemPromptExtra;
      return extra ? base + extra : base;
    },

    openThemeSelector() {
      this.showThemeSelector = true;
    },

    async selectTheme(themeId) {
      if (themeId === this.currentThemeId) {
        this.showThemeSelector = false;
        return;
      }
      const theme = this.conversationThemes.find(t => t.id === themeId);
      if (!theme || !theme.free) return;

      this.currentThemeId = themeId;
      this.showThemeSelector = false;
      this._persistConversationTheme();
      this._applyCurrentTheme();

      this.messages = [];
      this.showWelcome = true;
      try {
        localStorage.removeItem(this.STORAGE_KEY);
      } catch (e) {
        // localStorage unavailable
      }

      this.refreshHints();
      await this._recreateSession();
    },

    async _recreateSession() {
      if (this.session) {
        try { this.session.destroy(); } catch (e) { /* ignore */ }
        this.session = null;
      }
      this.statusText = "じゅんびちゅう...";
      this.statusType = "";
      await this._initApi();
    },

    _persistConversationTheme() {
      try {
        localStorage.setItem(this.THEME_KEY, this.currentThemeId);
      } catch (e) {
        // localStorage unavailable
      }
    },

    _restoreConversationTheme() {
      try {
        const saved = localStorage.getItem(this.THEME_KEY);
        if (saved && this.conversationThemes.some(t => t.id === saved)) {
          this.currentThemeId = saved;
        }
      } catch (e) {
        // localStorage unavailable
      }
    },

    // --- Text cleanup ---
    cleanText(text) {
      return text
        .replace(/[*#_~`>|]/g, "")
        .replace(/^-\s+/gm, "")
        .replace(/^\d+\.\s+/gm, "")
        .replace(/\n{2,}/g, "\n")
        // Remove verbose preamble phrases from Gemini
        .replace(/^(はい[、。！]?\s*|承知(いた)?しました[。！]?\s*|かしこまりました[。！]?\s*|もちろん(です)?[。！]?\s*|わかりました[。！]?\s*|いいですね[。！]?\s*|喜んで[。！]?\s*|了解(です|しました)?[。！]?\s*|それでは[、。]?\s*|では[、。]?\s*)+/i, "")
        // Remove topic echo like "〜ですね。" or "〜ですね！"
        .replace(/^[^。！？!?\n]{0,30}ですね[。！]?\s*/, "")
        .trim();
    },

    // --- UI utilities ---
    _isNearBottom() {
      const el = this.$refs.chatMessages;
      if (!el) return true;
      const threshold = 80;
      return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    },

    scrollToBottom() {
      this.$nextTick(() => {
        requestAnimationFrame(() => {
          const el = this.$refs.chatMessages;
          if (!el) return;
          if (!this._isNearBottom()) return;
          el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
        });
      });
    },

    resizeTextarea() {
      const el = this.$refs.messageInput;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 120) + "px";
    },

    // --- Text-to-speech ---
    speakText(text) {
      if (!window.speechSynthesis) return;
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "ja-JP";
      utterance.rate = 0.9;
      utterance.pitch = 1.1;
      utterance.onend = () => {
        if (this.voiceMode) {
          this._startListening();
        }
      };
      speechSynthesis.speak(utterance);
    },

    // --- Audio feedback ---
    playTone(freq, duration) {
      try {
        if (!this._audioCtx) this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = this._audioCtx.createOscillator();
        const gain = this._audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.value = 0.15;
        gain.gain.exponentialRampToValueAtTime(0.001, this._audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(this._audioCtx.destination);
        osc.start();
        osc.stop(this._audioCtx.currentTime + duration);
      } catch (e) { /* audio not available */ }
    },

    playMicOnSound() {
      this.playTone(880, 0.12);
      setTimeout(() => this.playTone(1175, 0.15), 100);
    },

    playMicOffSound() {
      this.playTone(784, 0.12);
      setTimeout(() => this.playTone(587, 0.15), 100);
    },

    // --- Usage tracking ---
    _setupUsageTracking() {
      this._purgeOldUsage();
      window.addEventListener("beforeunload", () => {
        this._recordSession();
      });
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") {
          this._recordSession();
        }
      });
    },

    _recordSession() {
      if (!this._sessionStartTime) return;
      const duration = Math.round((Date.now() - this._sessionStartTime) / 1000);
      if (duration < 5) return;
      try {
        const log = JSON.parse(localStorage.getItem(this._profileStorageKey(this.USAGE_KEY)) || "[]");
        const today = new Date().toISOString().slice(0, 10);
        log.push({ date: today, duration: duration, timestamp: Date.now() });
        localStorage.setItem(this._profileStorageKey(this.USAGE_KEY), JSON.stringify(log));
      } catch (e) {
        console.error("Failed to record usage:", e);
      }
      this._sessionStartTime = Date.now();
    },

    _purgeOldUsage() {
      try {
        const log = JSON.parse(localStorage.getItem(this._profileStorageKey(this.USAGE_KEY)) || "[]");
        const cutoff = Date.now() - this.USAGE_RETENTION_DAYS * 24 * 60 * 60 * 1000;
        const filtered = log.filter((entry) => entry.timestamp > cutoff);
        if (filtered.length !== log.length) {
          localStorage.setItem(this._profileStorageKey(this.USAGE_KEY), JSON.stringify(filtered));
        }
      } catch (e) {
        console.error("Failed to purge old usage:", e);
      }
    },

    // --- Dashboard ---
    openDashboard() {
      this._recordSession();
      this.dashboardData = this._buildDashboardData();
      this.showDashboard = true;
    },

    closeDashboard() {
      this.showDashboard = false;
      this.dashboardData = null;
    },

    _buildDashboardData() {
      const log = JSON.parse(localStorage.getItem(this._profileStorageKey(this.USAGE_KEY)) || "[]");
      const history = JSON.parse(localStorage.getItem(this._profileStorageKey(this.STORAGE_KEY)) || "[]");

      const totalSeconds = log.reduce((sum, e) => sum + e.duration, 0);
      const sessionCount = log.length;

      const dailyMap = {};
      log.forEach((e) => {
        dailyMap[e.date] = (dailyMap[e.date] || 0) + e.duration;
      });

      const today = new Date();
      const dailyUsage = [];
      for (let i = 6; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const key = d.toISOString().slice(0, 10);
        const mins = Math.round((dailyMap[key] || 0) / 60);
        const dayLabel = ["にち", "げつ", "か", "すい", "もく", "きん", "ど"][d.getDay()];
        dailyUsage.push({ date: key, dayLabel: dayLabel, minutes: mins });
      }

      // Weekly aggregation (past 4 weeks)
      const weeklyUsage = [];
      for (let w = 3; w >= 0; w--) {
        const weekStart = new Date(today);
        weekStart.setDate(weekStart.getDate() - weekStart.getDay() - (w * 7));
        let weekTotal = 0;
        for (let d = 0; d < 7; d++) {
          const dd = new Date(weekStart);
          dd.setDate(dd.getDate() + d);
          const key = dd.toISOString().slice(0, 10);
          weekTotal += (dailyMap[key] || 0);
        }
        const weekLabel = (w === 0) ? "こんしゅう" : w + "しゅうまえ";
        weeklyUsage.push({ label: weekLabel, minutes: Math.round(weekTotal / 60) });
      }

      const averageDailyMinutes = dailyUsage.length > 0
        ? Math.round(dailyUsage.reduce((s, d) => s + d.minutes, 0) / dailyUsage.length)
        : 0;

      const longestSession = log.length > 0
        ? Math.round(Math.max(...log.map((e) => e.duration)) / 60)
        : 0;

      const topics = this._extractTopics(history);

      return {
        totalMinutes: Math.round(totalSeconds / 60),
        sessionCount: sessionCount,
        dailyUsage: dailyUsage,
        weeklyUsage: weeklyUsage,
        averageDailyMinutes: averageDailyMinutes,
        longestSession: longestSession,
        topics: topics,
        stampCount: this.earnedStamps.length,
        stampTotal: this.stamps.length,
        activeProfile: this.activeProfile,
      };
    },

    // --- Premium ---
    _initPremium() {
      try {
        this.premiumUnlocked = localStorage.getItem(this.PREMIUM_KEY) === "true";
      } catch (e) {
        this.premiumUnlocked = false;
      }
    },

    togglePremium() {
      this.premiumUnlocked = !this.premiumUnlocked;
      try {
        localStorage.setItem(this.PREMIUM_KEY, String(this.premiumUnlocked));
      } catch (e) {
        // localStorage unavailable
      }
    },

    // --- Profiles ---
    _initProfiles() {
      try {
        const saved = localStorage.getItem(this.PROFILES_KEY);
        if (saved) {
          this.profiles = JSON.parse(saved);
        }
        if (this.profiles.length === 0) {
          this.profiles = [{ id: "default", name: "デフォルト", icon: "bear", age: "" }];
          this._saveProfiles();
        }
        const activeId = localStorage.getItem(this.ACTIVE_PROFILE_KEY);
        if (activeId && this.profiles.some((p) => p.id === activeId)) {
          this.activeProfileId = activeId;
        } else {
          this.activeProfileId = this.profiles[0].id;
        }
      } catch (e) {
        this.profiles = [{ id: "default", name: "デフォルト", icon: "bear", age: "" }];
        this.activeProfileId = "default";
      }
    },

    _saveProfiles() {
      try {
        localStorage.setItem(this.PROFILES_KEY, JSON.stringify(this.profiles));
      } catch (e) {
        // localStorage unavailable
      }
    },

    _profileStorageKey(baseKey) {
      if (this.activeProfileId === "default") return baseKey;
      return baseKey + "_" + this.activeProfileId;
    },

    get activeProfile() {
      return this.profiles.find((p) => p.id === this.activeProfileId) || this.profiles[0];
    },

    get activeProfileIcon() {
      const profile = this.activeProfile;
      const iconDef = this.PROFILE_ICONS.find((i) => i.id === profile.icon);
      return iconDef ? iconDef.emoji : "\uD83D\uDC3B";
    },

    openProfileManager() {
      this.showProfileManager = true;
      this.showProfileForm = false;
      this.editingProfile = null;
    },

    closeProfileManager() {
      this.showProfileManager = false;
      this.showProfileForm = false;
      this.editingProfile = null;
    },

    openProfileForm(profile) {
      if (profile) {
        this.editingProfile = profile.id;
        this.profileForm = { name: profile.name, icon: profile.icon, age: profile.age || "" };
      } else {
        if (this.profiles.length >= this.MAX_PROFILES) return;
        this.editingProfile = null;
        this.profileForm = { name: "", icon: "bear", age: "" };
      }
      this.showProfileForm = true;
    },

    saveProfile() {
      const name = this.profileForm.name.trim();
      if (!name) return;
      if (this.editingProfile) {
        const profile = this.profiles.find((p) => p.id === this.editingProfile);
        if (profile) {
          profile.name = name;
          profile.icon = this.profileForm.icon;
          profile.age = this.profileForm.age;
        }
      } else {
        const id = "profile_" + Date.now();
        this.profiles.push({ id: id, name: name, icon: this.profileForm.icon, age: this.profileForm.age });
      }
      this._saveProfiles();
      this.showProfileForm = false;
      this.editingProfile = null;
    },

    deleteProfile(profileId) {
      if (profileId === "default") return;
      if (this.profiles.length <= 1) return;
      this.profiles = this.profiles.filter((p) => p.id !== profileId);
      if (this.activeProfileId === profileId) {
        this.switchProfile(this.profiles[0].id);
      }
      this._saveProfiles();
      // Clean up profile-specific storage
      try {
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key && key.endsWith("_" + profileId)) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach((k) => localStorage.removeItem(k));
      } catch (e) {
        // localStorage unavailable
      }
    },

    switchProfile(profileId) {
      const profile = this.profiles.find((p) => p.id === profileId);
      if (!profile) return;
      // Save current session before switching
      this._recordSession();
      this._saveHistory();
      // Switch
      this.activeProfileId = profileId;
      try {
        localStorage.setItem(this.ACTIVE_PROFILE_KEY, profileId);
      } catch (e) {
        // localStorage unavailable
      }
      // Reload profile data
      this._restoreHistory();
      this._sessionStartTime = Date.now();
      if (this.showDashboard) {
        this.dashboardData = this._buildDashboardData();
      }
      this.showProfileManager = false;
    },

    // --- Data migration ---
    openMigration() {
      if (!this._parentalUnlocked) {
        this.openParentalSettings();
        return;
      }
      this.migrationStatus = "";
      this.migrationError = "";
      this.showMigrationOverlay = true;
    },

    closeMigration() {
      this.showMigrationOverlay = false;
      this.migrationStatus = "";
      this.migrationError = "";
    },

    exportData() {
      try {
        const data = {};
        this.MIGRATION_KEYS.forEach((key) => {
          const value = localStorage.getItem(key);
          if (value !== null) {
            data[key] = value;
          }
        });
        const exportObj = {
          version: this.EXPORT_VERSION,
          exportedAt: new Date().toISOString(),
          data: data,
        };
        const json = JSON.stringify(exportObj, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = this.EXPORT_FILENAME;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.migrationStatus = "ファイルに ほぞんしたよ！";
        this.migrationError = "";
      } catch (e) {
        console.error("Export failed:", e);
        this.migrationError = "エクスポートに しっぱいしたよ";
        this.migrationStatus = "";
      }
    },

    importData() {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json";
      input.style.display = "none";
      input.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (file.size > this.MIGRATION_MAX_SIZE) {
          console.warn("Import file is larger than 5MB:", file.size);
        }
        const reader = new FileReader();
        reader.onload = (ev) => {
          try {
            const parsed = JSON.parse(ev.target.result);
            if (!parsed.version) {
              this.migrationError = "ファイルの けいしきが ただしくないよ";
              this.migrationStatus = "";
              return;
            }
            if (parsed.version > this.EXPORT_VERSION) {
              this.migrationError = "あたらしい バージョンの ファイルだよ。アプリを こうしんしてね";
              this.migrationStatus = "";
              return;
            }
            if (!parsed.data || typeof parsed.data !== "object") {
              this.migrationError = "ファイルに データが ないよ";
              this.migrationStatus = "";
              return;
            }
            Object.keys(parsed.data).forEach((key) => {
              if (this.MIGRATION_KEYS.includes(key)) {
                localStorage.setItem(key, parsed.data[key]);
              }
            });
            location.reload();
          } catch (err) {
            console.error("Import failed:", err);
            this.migrationError = "ファイルを よみこめなかったよ";
            this.migrationStatus = "";
          }
        };
        reader.readAsText(file);
      });
      document.body.appendChild(input);
      input.click();
      document.body.removeChild(input);
    },

    // --- Learning mode ---
    openLearningSelect() {
      this.showLearningSelect = true;
      this.learningCategory = null;
      this.learningLevel = null;
    },

    closeLearningSelect() {
      this.showLearningSelect = false;
    },

    selectLearningCategory(categoryId) {
      this.learningCategory = categoryId;
    },

    async startLearningMode(levelId) {
      this.learningLevel = levelId;
      this.showLearningSelect = false;

      if (!this.session) return;

      const api =
        typeof LanguageModel !== "undefined" ? LanguageModel :
        self.ai?.languageModel ? self.ai.languageModel :
        null;

      if (!api) return;

      const languageOptions = {
        expectedInputs: [{ type: "text", languages: ["ja"] }],
        expectedOutputs: [{ type: "text", languages: ["ja"] }],
      };

      const systemPrompt = this._buildLearningPrompt(this.learningCategory, this.learningLevel);

      try {
        this.learningSession = await api.create({
          ...languageOptions,
          systemPrompt: systemPrompt,
        });
      } catch (err) {
        console.error("Learning session creation failed:", err);
        return;
      }

      this.learningMode = true;
      this.messages = [];
      this.showWelcome = false;

      this.messages.push({
        id: this._nextId++,
        role: "ai",
        text: "",
        isTyping: true,
      });
      this.scrollToBottom();
      this._setMascotState("thinking");

      try {
        const stream = await this.learningSession.promptStreaming("もんだいを だして");
        this.messages[0].isTyping = false;
        this._setMascotState("talking");
        let fullText = "";
        for await (const chunk of stream) {
          fullText += chunk;
          this.messages[0].text = this.cleanText(fullText);
          this.scrollToBottom();
        }
        this._showHappyThenIdle();
        this.speakText(this.cleanText(fullText));
      } catch (err) {
        console.error("Learning prompt error:", err);
        this.messages[0].isTyping = false;
        this.messages[0].text = "ごめんね、もんだいを だせなかったよ。もういちど やってみてね。";
        this._setMascotState("idle");
      }
    },

    async endLearningMode() {
      if (this.learningCategory) {
        this.recordLearningSession(this.learningCategory);
      }
      this.learningMode = false;
      this.learningSession = null;
      this.learningCategory = null;
      this.learningLevel = null;
      this.messages = [];
      this.showWelcome = true;
      this.refreshHints();
    },

    _buildLearningPrompt(category, level) {
      const base =
        "あなたは子ども向けの学習アシスタントです。やさしい日本語で、ひらがなを使って話してください。" +
        "Markdownや記号は絶対に使わないでください。" +
        "危険なこと、暴力、いじめ、薬物、性的な内容には絶対に答えないでください。";

      const levelDesc = {
        easy: "とてもかんたんなレベル。4〜5さい向け。",
        normal: "ふつうのレベル。6〜7さい向け。",
        hard: "すこしむずかしいレベル。7〜8さい向け。",
      };

      const prompts = {
        hiragana:
          base +
          "あなたはひらがなの先生です。" + levelDesc[level] +
          "ひらがなの読み書きや、ことばの練習をします。" +
          "【ルール】1もんだいずつ出してください。こたえを聞いたら正解かどうか教えて、ほめてあげてください。" +
          "まちがえたらやさしくヒントを出してください。" +
          "「もんだいを だして」と言われたら、すぐに問題だけを出してください。前置きは不要です。" +
          (level === "easy"
            ? "「あ」〜「ん」の文字を見せて読ませる問題や、2〜3文字のかんたんなことばの問題を出してください。"
            : level === "normal"
            ? "3〜4文字のことばの問題や、ことばの穴埋め問題を出してください。"
            : "4〜5文字のことばや、文をつくる問題を出してください。"),
        math:
          base +
          "あなたはさんすうの先生です。" + levelDesc[level] +
          "たしざんやひきざんの練習をします。" +
          "【ルール】1もんだいずつ出してください。こたえを聞いたら正解かどうか教えて、ほめてあげてください。" +
          "まちがえたらやさしくヒントを出してください。" +
          "「もんだいを だして」と言われたら、すぐに問題だけを出してください。前置きは不要です。" +
          (level === "easy"
            ? "1〜5までの たしざんだけ出してください。例: 「1 たす 2 は？」"
            : level === "normal"
            ? "1〜10までの たしざんと ひきざんを出してください。例: 「7 たす 3 は？」「8 ひく 2 は？」"
            : "10〜20までの たしざんと ひきざんを出してください。2けたの計算もOKです。"),
        english:
          base +
          "あなたは英語の先生です。" + levelDesc[level] +
          "かんたんな英単語のクイズをします。" +
          "【ルール】1もんだいずつ出してください。こたえを聞いたら正解かどうか教えて、ほめてあげてください。" +
          "まちがえたらやさしくヒントを出してください。" +
          "「もんだいを だして」と言われたら、すぐに問題だけを出してください。前置きは不要です。" +
          (level === "easy"
            ? "どうぶつやくだもの、いろの英単語を出してください。例: 「りんごは えいごで なに？」"
            : level === "normal"
            ? "からだのぶぶん、かぞく、たべものの英単語を出してください。例: 「あたまは えいごで なに？」"
            : "きせつ、ようび、きもちの英単語を出してください。例: 「はるは えいごで なに？」"),
      };
      return prompts[category] || base;
    },

    _restoreLearningProgress() {
      try {
        const data = localStorage.getItem(this.LEARNING_PROGRESS_KEY);
        if (data) {
          this.learningProgress = JSON.parse(data);
        }
      } catch (e) {
        this.learningProgress = {};
      }
    },

    _saveLearningProgress() {
      try {
        localStorage.setItem(this.LEARNING_PROGRESS_KEY, JSON.stringify(this.learningProgress));
      } catch (e) {
        console.error("Failed to save learning progress:", e);
      }
    },

    recordLearningSession(category) {
      if (!this.learningProgress[category]) {
        this.learningProgress[category] = { sessions: 0 };
      }
      this.learningProgress[category].sessions++;
      this._saveLearningProgress();
    },

    getLearningProgressFor(categoryId) {
      const p = this.learningProgress[categoryId];
      return p ? p.sessions + "かい" : "0かい";
    },

    // --- Stamp / reward system ---
    _initStamps() {
      this.stamps = this.STAMP_DEFINITIONS.map((def) => ({ ...def }));
      try {
        const saved = JSON.parse(localStorage.getItem(this.STAMP_KEY) || "[]");
        this.earnedStamps = Array.isArray(saved) ? saved : [];
      } catch (e) {
        this.earnedStamps = [];
      }
      this._categoriesUsed = new Set([this.selectedCategory]);
      // Count existing chat messages for stamp progress
      try {
        const history = JSON.parse(localStorage.getItem(this.STORAGE_KEY) || "[]");
        this._chatCount = history.filter((m) => m.role === "user").length;
      } catch (e) {
        this._chatCount = 0;
      }
    },

    _getConsecutiveDays() {
      try {
        const log = JSON.parse(localStorage.getItem(this.USAGE_KEY) || "[]");
        if (log.length === 0) return 0;
        const dates = new Set(log.map((e) => e.date));
        const today = new Date().toISOString().slice(0, 10);
        if (!dates.has(today)) {
          // Include today since we're in a session
          dates.add(today);
        }
        let streak = 0;
        const d = new Date();
        while (true) {
          const key = d.toISOString().slice(0, 10);
          if (dates.has(key)) {
            streak++;
            d.setDate(d.getDate() - 1);
          } else {
            break;
          }
        }
        return streak;
      } catch (e) {
        return 0;
      }
    },

    _checkStamps() {
      const earned = new Set(this.earnedStamps.map((s) => s.id));
      const streak = this._getConsecutiveDays();
      let newStamp = null;

      for (const stamp of this.stamps) {
        if (earned.has(stamp.id)) continue;
        let achieved = false;
        if (stamp.condition === "chat") {
          achieved = this._chatCount >= stamp.threshold;
        } else if (stamp.condition === "streak") {
          achieved = streak >= stamp.threshold;
        } else if (stamp.condition === "voice") {
          achieved = this._voiceUsedCount >= stamp.threshold;
        } else if (stamp.condition === "theme") {
          achieved = true; // triggered only on theme toggle
        } else if (stamp.condition === "category") {
          achieved = this._categoriesUsed && this._categoriesUsed.size >= stamp.threshold;
        }
        if (achieved) {
          const entry = { id: stamp.id, earnedAt: new Date().toISOString() };
          this.earnedStamps.push(entry);
          earned.add(stamp.id);
          newStamp = stamp;
        }
      }

      if (newStamp) {
        this._saveStamps();
        this._showStampEarned(newStamp);
      }
    },

    _saveStamps() {
      try {
        localStorage.setItem(this.STAMP_KEY, JSON.stringify(this.earnedStamps));
      } catch (e) {
        // localStorage unavailable
      }
    },

    _showStampEarned(stamp) {
      this.earnedStampName = stamp.name;
      this.earnedStampIcon = stamp.icon;
      this.showStampEarned = true;
      this.playStampSound();
      if (this._stampEarnedTimer) clearTimeout(this._stampEarnedTimer);
      this._stampEarnedTimer = setTimeout(() => {
        this.showStampEarned = false;
        this._stampEarnedTimer = null;
      }, 3000);
    },

    dismissStampEarned() {
      this.showStampEarned = false;
      if (this._stampEarnedTimer) {
        clearTimeout(this._stampEarnedTimer);
        this._stampEarnedTimer = null;
      }
    },

    openStampCollection() {
      this.showStampCollection = true;
    },

    closeStampCollection() {
      this.showStampCollection = false;
    },

    isStampEarned(stampId) {
      return this.earnedStamps.some((s) => s.id === stampId);
    },

    get stampEarnedCount() {
      return this.earnedStamps.length;
    },

    playStampSound() {
      this.playTone(523, 0.1);
      setTimeout(() => this.playTone(659, 0.1), 80);
      setTimeout(() => this.playTone(784, 0.15), 160);
      setTimeout(() => this.playTone(1047, 0.2), 240);
    },

    // --- Content packs ---
    async _initContentPacks() {
      try {
        const saved = localStorage.getItem(this.PACKS_STORAGE_KEY);
        if (saved) {
          this.installedPacks = JSON.parse(saved);
        }
        const savedActive = localStorage.getItem(this.ACTIVE_PACK_KEY);
        if (savedActive && this.installedPacks.some((p) => p.id === savedActive)) {
          this.activePackId = savedActive;
        }
      } catch (e) {
        this.installedPacks = [];
        this.activePackId = null;
      }
    },

    async openPackStore() {
      this.showPackStore = true;
      this.packLoading = true;
      try {
        const res = await fetch(this.PACK_MANIFEST_URL);
        if (!res.ok) throw new Error("Failed to load manifest");
        const manifest = await res.json();
        const packs = [];
        for (const file of manifest.packs) {
          try {
            const packRes = await fetch("./packs/" + file);
            if (packRes.ok) {
              const pack = await packRes.json();
              if (this._validatePack(pack)) {
                packs.push(pack);
              }
            }
          } catch (e) {
            console.error("Failed to load pack:", file, e);
          }
        }
        this.availablePacks = packs;
      } catch (e) {
        console.error("Failed to load pack manifest:", e);
        this.availablePacks = [];
      } finally {
        this.packLoading = false;
      }
    },

    closePackStore() {
      this.showPackStore = false;
    },

    _validatePack(pack) {
      return (
        pack &&
        typeof pack.id === "string" &&
        typeof pack.name === "string" &&
        typeof pack.version === "string" &&
        typeof pack.type === "string" &&
        typeof pack.free === "boolean" &&
        pack.content &&
        typeof pack.content === "object"
      );
    },

    isPackInstalled(packId) {
      return this.installedPacks.some((p) => p.id === packId);
    },

    installPack(pack) {
      if (this.isPackInstalled(pack.id)) return;
      if (!pack.free) return;
      this.installedPacks.push({
        id: pack.id,
        name: pack.name,
        version: pack.version,
        type: pack.type,
        icon: pack.icon || "",
        content: pack.content,
      });
      this._saveInstalledPacks();
    },

    uninstallPack(packId) {
      this.installedPacks = this.installedPacks.filter((p) => p.id !== packId);
      if (this.activePackId === packId) {
        this.activePackId = null;
        localStorage.removeItem(this.ACTIVE_PACK_KEY);
      }
      this._saveInstalledPacks();
    },

    activatePack(packId) {
      if (!this.isPackInstalled(packId)) return;
      if (this.activePackId === packId) {
        this.activePackId = null;
        localStorage.removeItem(this.ACTIVE_PACK_KEY);
      } else {
        this.activePackId = packId;
        try {
          localStorage.setItem(this.ACTIVE_PACK_KEY, packId);
        } catch (e) {
          // localStorage unavailable
        }
      }
    },

    getActivePack() {
      if (!this.activePackId) return null;
      return this.installedPacks.find((p) => p.id === this.activePackId) || null;
    },

    getActivePackHints() {
      const pack = this.getActivePack();
      if (!pack || !pack.content || !pack.content.hints) return [];
      return pack.content.hints;
    },

    _saveInstalledPacks() {
      try {
        localStorage.setItem(this.PACKS_STORAGE_KEY, JSON.stringify(this.installedPacks));
      } catch (e) {
        console.error("Failed to save installed packs:", e);
      }
    },

    // --- Payment system ---
    _initPayment() {
      try {
        const saved = localStorage.getItem(this.PURCHASES_KEY);
        if (saved) {
          const parsed = JSON.parse(saved);
          if (parsed && typeof parsed === "object") {
            this.purchasedItems = parsed;
          }
        }
      } catch (e) {
        this.purchasedItems = {};
      }
    },

    isPurchased(itemId) {
      return !!this.purchasedItems[itemId];
    },

    _savePurchases() {
      try {
        localStorage.setItem(this.PURCHASES_KEY, JSON.stringify(this.purchasedItems));
      } catch (e) {
        console.error("Failed to save purchases:", e);
      }
    },

    startPurchase(item) {
      if (!this._parentalUnlocked) {
        this._pendingPurchaseItem = item;
        this._purchaseAfterUnlock = true;
        this.openParentalSettings();
        return;
      }
      this.paymentItem = item;
      this.paymentStep = "confirm";
      this.paymentError = "";
      this.showPaymentOverlay = true;
    },

    confirmPurchase() {
      if (!this.paymentItem) return;
      this.paymentStep = "processing";
      this.paymentError = "";

      // ExtensionPay integration point: in production, this calls ExtPay.openPaymentPage()
      // For now, simulate the payment flow with a configurable handler
      this._processPayment(this.paymentItem);
    },

    _processPayment(item) {
      // Check if ExtensionPay is available (Chrome extension environment)
      if (typeof ExtPay !== "undefined") {
        try {
          ExtPay("vibekid").openPaymentPage();
          // ExtensionPay handles the rest via its callback
          // For MVP, we listen for the payment completion via storage change
          this._waitForExtPayCompletion(item);
        } catch (e) {
          this.paymentStep = "error";
          this.paymentError = "けっさいサービスに つながらなかったよ";
        }
        return;
      }

      // Fallback: mark as payment service unavailable
      // In production build, ExtensionPay SDK will be bundled
      this.paymentStep = "error";
      this.paymentError = "けっさいサービスが まだ つかえないよ。あとで もういちど ためしてね";
    },

    _waitForExtPayCompletion(item) {
      // ExtensionPay uses chrome.storage to signal payment completion
      // Poll for a reasonable time, then timeout
      const startTime = Date.now();
      const maxWait = 300000; // 5 minutes
      const checkInterval = setInterval(() => {
        if (Date.now() - startTime > maxWait) {
          clearInterval(checkInterval);
          this.paymentStep = "error";
          this.paymentError = "じかんが かかりすぎたよ。もういちど ためしてね";
          return;
        }
        // Check if ExtPay has updated the user's paid status
        try {
          ExtPay("vibekid").getUser().then((user) => {
            if (user.paid) {
              clearInterval(checkInterval);
              this._completePurchase(item);
            }
          });
        } catch (e) {
          clearInterval(checkInterval);
          this.paymentStep = "error";
          this.paymentError = "けっさいの かくにんに しっぱいしたよ";
        }
      }, 3000);
    },

    _completePurchase(item) {
      this.purchasedItems[item.id] = {
        purchasedAt: new Date().toISOString(),
        name: item.name,
        type: item.type,
      };
      this._savePurchases();
      this._recordPurchase(item);
      this.paymentStep = "success";
    },

    cancelPurchase() {
      this.showPaymentOverlay = false;
      this.paymentItem = null;
      this.paymentStep = "confirm";
      this.paymentError = "";
    },

    closePurchaseSuccess() {
      this.showPaymentOverlay = false;
      this.paymentItem = null;
      this.paymentStep = "confirm";
    },

    openRestorePurchases() {
      if (!this._parentalUnlocked) {
        this.openParentalSettings();
        return;
      }
      this.restoreStatus = "";
      this.restoreError = "";
      this.showRestoreOverlay = true;
    },

    closeRestorePurchases() {
      this.showRestoreOverlay = false;
      this.restoreStatus = "";
      this.restoreError = "";
    },

    restorePurchases() {
      this.restoreStatus = "";
      this.restoreError = "";

      if (typeof ExtPay !== "undefined") {
        try {
          ExtPay("vibekid").getUser().then((user) => {
            if (user.paid) {
              // Restore all content for paid users
              // In production, query specific entitlements
              this.restoreStatus = "こうにゅうずみの コンテンツを ふっきしたよ！";
              this.restoreError = "";
            } else {
              this.restoreStatus = "こうにゅうずみの コンテンツは みつからなかったよ";
            }
          }).catch(() => {
            this.restoreError = "ふっきに しっぱいしたよ。あとで もういちど ためしてね";
            this.restoreStatus = "";
          });
        } catch (e) {
          this.restoreError = "けっさいサービスに つながらなかったよ";
          this.restoreStatus = "";
        }
        return;
      }

      this.restoreError = "けっさいサービスが まだ つかえないよ";
      this.restoreStatus = "";
    },

    getPurchasedCount() {
      return Object.keys(this.purchasedItems).length;
    },

    _extractTopics(history) {
      const keywords = {};
      const stopWords = new Set([
        "して", "ついて", "おしえて", "なに", "どう", "ある", "いる", "する", "できる",
        "ない", "ほしい", "たい", "って", "から", "まで", "もう", "また", "ちょっと",
        "すごい", "いい", "ある", "これ", "それ", "あれ", "だれ", "どこ", "いつ",
        "なんで", "なぜ", "どれ", "だった", "です", "ます", "ました",
      ]);
      const suffixPattern = /(?:っていう|について|っている|ってる|という|ている|てる|では|には|とは|への|だけ|より|って|から|まで|だね|かな|のに|けど|ので|たち|など|は|を|の|が|に|で|も|と|へ|よ|ね|な|か|だ|さ)+$/;

      const userMessages = history.filter((m) => m.role === "user");
      userMessages.forEach((m) => {
        const text = m.text || "";
        const words = text.replace(/[、。！？!?\s]+/g, " ").trim().split(" ");
        words.forEach((w) => {
          const cleaned = w.replace(suffixPattern, "");
          if (cleaned.length >= 2 && !stopWords.has(cleaned)) {
            keywords[cleaned] = (keywords[cleaned] || 0) + 1;
          }
        });
      });

      return Object.entries(keywords)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([word, count]) => ({ word, count }));
    },

    // --- Store ---
    openStore() {
      this.showStore = true;
      this.storeTab = "packs";
      this._recordStoreView();
    },

    closeStore() {
      this.showStore = false;
    },

    setStoreTab(tab) {
      this.storeTab = tab;
    },

    _recordStoreView() {
      try {
        const metrics = JSON.parse(localStorage.getItem(this.STORE_METRICS_KEY) || "{}");
        metrics.views = (metrics.views || 0) + 1;
        localStorage.setItem(this.STORE_METRICS_KEY, JSON.stringify(metrics));
      } catch (e) {
        // metrics recording is best-effort
      }
    },

    _recordPurchase(item) {
      try {
        const metrics = JSON.parse(localStorage.getItem(this.STORE_METRICS_KEY) || "{}");
        metrics.purchases = (metrics.purchases || 0) + 1;
        if (!metrics.purchaseLog) metrics.purchaseLog = [];
        metrics.purchaseLog.push({
          id: item.id,
          type: item.type,
          price: item.price,
          timestamp: Date.now(),
        });
        localStorage.setItem(this.STORE_METRICS_KEY, JSON.stringify(metrics));
      } catch (e) {
        // metrics recording is best-effort
      }
    },

    get storeItems() {
      if (this.storeTab === "packs") return this.paidContentPacks;
      return this.paidKisekaeItems;
    },
  }));
});
