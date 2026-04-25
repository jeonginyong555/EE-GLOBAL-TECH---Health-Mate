const dom = {
  body: document.body,
  modeScreen: document.getElementById("mode-screen"),
  appScreen: document.getElementById("app-screen"),
  themeToggle: document.getElementById("themeToggle"),
  backBtn: document.getElementById("backBtn"),
  modeCards: document.querySelectorAll(".mode-card"),

  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  overlay: document.getElementById("overlay"),
  threeWrap: document.getElementById("threeWrap"),
  guideCard: document.getElementById("guideCard"),
  guideVideo: document.getElementById("guideVideo"),

  modeChip: document.getElementById("modeChip"),
  statusChip: document.getElementById("statusChip"),
  helpChip: document.getElementById("helpChip"),
  gestureChip: document.getElementById("gestureChip"),

  infoMain: document.getElementById("infoMain"),
  infoSub: document.getElementById("infoSub"),

  feedbackMain: document.getElementById("feedbackMain"),
  feedbackDetail: document.getElementById("feedbackDetail"),
  feedbackStateText: document.getElementById("feedbackStateText"),
  feedbackPoseText: document.getElementById("feedbackPoseText"),
  feedbackGuideText: document.getElementById("feedbackGuideText"),
  analysisStatus: document.getElementById("analysisStatus"),

  debugBox: document.getElementById("debugBox"),
  manualToggleBtn: document.getElementById("manualToggleBtn"),
  restartBtn: document.getElementById("restartBtn"),
  cameraRotateToggle: document.getElementById("cameraRotateToggle"),
};

const state = {
  currentMode: null,
  currentController: null,
  themeLight: false,
};

function setTheme() {
  state.themeLight = !state.themeLight;
  dom.body.classList.toggle("light-mode", state.themeLight);
  dom.themeToggle.textContent = state.themeLight ? "☀️ LIGHT MODE" : "🌙 DARK MODE";
}

function showModeScreen() {
  dom.modeScreen.classList.remove("hidden");
  dom.appScreen.classList.add("hidden");
}

function showAppScreen() {
  dom.modeScreen.classList.add("hidden");
  dom.appScreen.classList.remove("hidden");
}

function resetViewVisibility() {
  dom.video.style.display = "none";
  dom.canvas.style.display = "none";
  dom.overlay.style.display = "none";
  dom.threeWrap.style.display = "none";
  dom.guideCard.style.display = "none";
}

function setStatus(text) {
  dom.statusChip.textContent = text;
  if (dom.analysisStatus) dom.analysisStatus.textContent = text;
}

function setInfo(title, desc) {
  dom.infoMain.textContent = title;
  dom.infoSub.textContent = desc;
}

function setHelp(text) {
  dom.helpChip.textContent = text;
}

function setDebug(...lines) {
  dom.debugBox.textContent = lines.join("\n");
}

function setFeedback({
  main = "준비 중",
  detail = "자세를 확인하고 있습니다.",
  state = "대기",
  pose = "화면 안으로 몸을 맞춰주세요.",
  guide = "전신 또는 주요 관절이 보이도록 위치를 조정해주세요."
}) {
  dom.feedbackMain.textContent = main;
  dom.feedbackDetail.textContent = detail;
  dom.feedbackStateText.textContent = state;
  dom.feedbackPoseText.textContent = pose;
  dom.feedbackGuideText.textContent = guide;
}

async function destroyCurrentMode() {
  if (state.currentController?.destroy) {
    await state.currentController.destroy();
  }
  state.currentController = null;
  state.currentMode = null;
  resetViewVisibility();
  dom.gestureChip.textContent = "GESTURE: -";
}

async function enterMode(mode) {
  await destroyCurrentMode();
  state.currentMode = mode;
  showAppScreen();

  let factory;
  if (mode === "standard") {
    factory = await import("./jsonMode.js");
  } else if (mode === "feedback") {
    factory = await import("./feedbackMode.js");
  } else {
    factory = await import("./avatarMode.js");
  }

  state.currentController = await factory.createMode({
    dom,
    shared: {
      setStatus,
      setInfo,
      setHelp,
      setDebug,
      setFeedback,
      resetViewVisibility,
      themeLight: () => state.themeLight,
    }
  });

  await state.currentController.start();
}

async function restartMode() {
  if (!state.currentMode) return;
  await enterMode(state.currentMode);
}

dom.themeToggle.addEventListener("click", setTheme);

dom.modeCards.forEach((card) => {
  card.addEventListener("click", async () => {
    const mode = card.dataset.mode;
    await enterMode(mode);
  });
});

dom.backBtn.addEventListener("click", async () => {
  await destroyCurrentMode();
  showModeScreen();
  setStatus("READY");
  setInfo("대기 중", "모드를 선택하세요.");
  setHelp("설정 대기 중");
});

dom.restartBtn.addEventListener("click", restartMode);

showModeScreen();
setStatus("READY");
setInfo("대기 중", "모드를 선택하세요.");
setHelp("설정 대기 중");
setFeedback({});
setDebug("MAIN READY");
console.log("MAIN READY");