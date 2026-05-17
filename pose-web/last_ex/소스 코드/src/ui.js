    (() => {
      const homePage = document.getElementById("homePage");
      const modePage = document.getElementById("modePage");
      const setupPage = document.getElementById("setupPage");
      const setupBody = document.getElementById("setupBody");

      const startWorkoutBtn = document.getElementById("startWorkoutBtn");
      const modeBackBtn = document.getElementById("modeBackBtn");
      const setupBackBtn = document.getElementById("setupBackBtn");

      const selectedModeChip = document.getElementById("selectedModeChip");
      const setupPageSubtitle = document.getElementById("setupPageSubtitle");

      const previewColumn = document.getElementById("previewColumn");
      const avatarPreviewBox = document.getElementById("avatarPreviewBox");
      const gymPreviewBox = document.getElementById("gymPreviewBox");
      const avatarPreviewImage = document.getElementById("avatarPreviewImage");
      const avatarPreviewName = document.getElementById("avatarPreviewName");
      const avatarPreviewCaption = document.getElementById("avatarPreviewCaption");
      const gymPreviewImage = document.getElementById("gymPreviewImage");
      const gymPreviewName = document.getElementById("gymPreviewName");

      const pages = { home: homePage, mode: modePage, setup: setupPage };

      const modeMeta = {
        standard: {
          label: "가이드 운동",
          subtitle: "스쿼트 기준 영상은 고정입니다. 아바타와 헬스장 배경만 선택합니다.",
          showAvatarPreview: true,
          showGymPreview: true,
        },
        feedback: {
          label: "자세 교정",
          subtitle: "영상과 웹캠을 바로 사용합니다. 별도 선택지는 없습니다.",
          showAvatarPreview: false,
          showGymPreview: false,
        },
        avatar: {
  label: "동작 따라하기",
  subtitle: "기존 아바타, 세트 수, 횟수, 휴식 시간을 설정합니다.",
  showAvatarPreview: true,
  showGymPreview: false,
},
retarget: {
  label: "자유 측정",
  subtitle: "리타게팅 아바타, 운동 선택, 세트 및 모드를 설정합니다.",
  showAvatarPreview: true,
  showGymPreview: false,
},
      };

      const avatarPreviewMap = {
        default: { name: "기본 RPM", img: "/images/avatar_default.png" },
        vroid1: { name: "VRoid 1", img: "/images/avatar_vroid1.png" },
        vroid2: { name: "VRoid 2", img: "/images/avatar_vroid2.png" },
        vroid3: { name: "VRoid 3", img: "/images/avatar_vroid3.png" },
        vroid4: { name: "VRoid 4", img: "/images/avatar_vroid4.png" },
      };

      const gymPreviewMap = {
        gym1: { name: "gym1", img: "/images/Untitled_gym.png" },
        gym2: { name: "gym2", img: "/images/Untitled_gym2.png" },
        gym3: { name: "gym3", img: "/images/Untitled_gym3.png" },
        gym4: { name: "gym4", img: "/images/Untitled_gym4.png" },
        cyber: { name: "Cyber Fitness", img: "/images/3d.png" },
        ptzone: { name: "PT Zone", img: "/images/astrogen.png" },
        darkroom: { name: "Dark Room", img: "/images/org.png" },
        brickgym: { name: "Brick Gym", img: "/images/teto.png" },
      };

      const retargetPreviewMap = {
        "model.glb|Mixamo": { name: "Avaturn Custom", img: "/images/model.png" },
        "boy.glb|Mixamo": { name: "Male / Mixamo", img: "/images/boy.png" },
        "girl.glb|Mixamo": { name: "Female / Mixamo", img: "/images/girl.png" },
        "main.glb|RPM": { name: "Main / RPM", img: "/images/main.png" },
      };

      let selectedMode = null;

      function updateRuntimePreviewPanel() {
  const panel = document.getElementById("runtimePreviewPanel");
  const avatarImg = document.getElementById("runtimePreviewAvatarImg");
  const avatarName = document.getElementById("runtimePreviewAvatarName");
  const gymImg = document.getElementById("runtimePreviewGymImg");
  const gymName = document.getElementById("runtimePreviewGymName");

  if (!panel || !avatarImg || !avatarName || !gymImg || !gymName) return;

  let avatarData = avatarPreviewMap.default;
  let gymData = gymPreviewMap.gym1;

  if (selectedMode === "retarget") {
    const avatarValue =
      document.getElementById("runtimeRetargetAvatarSelect")?.value ||
      document.getElementById("retargetAvatarSelect")?.value ||
      "model.glb|Mixamo";

    avatarData = retargetPreviewMap[avatarValue] || retargetPreviewMap["model.glb|Mixamo"];

    const gymValue =
      document.getElementById("runtimeRetargetGymSelect")?.value ||
      document.getElementById("retargetGymSelect")?.value ||
      "gym1";

    gymData = gymPreviewMap[gymValue] || gymPreviewMap.gym1;
  } else {
    const avatarValue =
      document.getElementById("runtimeAvatarSelect")?.value ||
      document.getElementById("avatarSelect")?.value ||
      "default";

    avatarData = avatarPreviewMap[avatarValue] || avatarPreviewMap.default;

    const gymValue =
      document.getElementById("runtimeGymSelect")?.value ||
      document.getElementById("gymSelect")?.value ||
      "gym1";

    gymData = gymPreviewMap[gymValue] || gymPreviewMap.gym1;
  }

  avatarImg.src = avatarData.img;
  avatarName.textContent = avatarData.name;

  gymImg.src = gymData.img;
  gymName.textContent = gymData.name;
}

document.getElementById("pipSettingsBtn")?.addEventListener("click", () => {
  setTimeout(() => {
    const settingsPanel = document.getElementById("pipSettingsPanel");
    const previewPanel = document.getElementById("runtimePreviewPanel");

    const isOpen = settingsPanel && !settingsPanel.classList.contains("hidden");

    previewPanel?.classList.toggle("hidden", !isOpen);

    if (isOpen) updateRuntimePreviewPanel();
  }, 0);
});

      function showPage(pageName) {
        Object.entries(pages).forEach(([name, el]) => {
          el.classList.toggle("hidden", name !== pageName);
        });
      }

      function hideAllSetupPanels() {
        document.querySelectorAll(".setup-panel").forEach((el) => el.classList.add("hidden"));
      }

      function showSetupForMode(mode) {
        selectedMode = mode;
        const meta = modeMeta[mode];

        selectedModeChip.textContent = `선택 모드: ${meta.label}`;
        setupPageSubtitle.textContent = meta.subtitle;

        hideAllSetupPanels();

        const panel = document.getElementById(`setup-${mode}`);
        if (panel) panel.classList.remove("hidden");

        const noPreview = !meta.showAvatarPreview && !meta.showGymPreview;
        setupBody.classList.toggle("no-preview", noPreview);
        previewColumn.classList.toggle("hidden", noPreview);
        avatarPreviewBox.classList.toggle("hidden", !meta.showAvatarPreview);
        gymPreviewBox.classList.toggle("hidden", !meta.showGymPreview);

        if (mode === "retarget") {
          updateRetargetPreview();
          avatarPreviewCaption.textContent = "현재 선택된 리타게팅 아바타";
          syncRetargetInputByExercise();
        } else {
          syncAvatarPreviewFromSelect(getCurrentAvatarSelect());
          avatarPreviewCaption.textContent = "현재 선택된 아바타 미리보기";
        }

        syncGymPreviewFromSelect(getCurrentGymSelect());
        showPage("setup");
      }

      function getCurrentAvatarSelect() {
        if (selectedMode === "avatar") return document.getElementById("avatarModeAvatarSelect");
        return document.getElementById("avatarSelect");
      }

      function getCurrentGymSelect() {
  if (selectedMode === "avatar") return document.getElementById("avatarModeGymSelect");
  if (selectedMode === "retarget") return document.getElementById("retargetGymSelect");
  return document.getElementById("gymSelect");
}

      function syncAvatarPreviewFromSelect(selectEl) {
        if (!selectEl) return;
        const data = avatarPreviewMap[selectEl.value] || avatarPreviewMap.default;
        avatarPreviewImage.src = data.img;
        avatarPreviewName.textContent = data.name;
      }

      function syncGymPreviewFromSelect(selectEl) {
        if (!selectEl) return;
        const data = gymPreviewMap[selectEl.value] || gymPreviewMap.gym1;
        gymPreviewImage.src = data.img;
        gymPreviewName.textContent = data.name;
      }

      function updateRetargetPreview() {
        const selectEl = document.getElementById("retargetAvatarSelect");
        const data = retargetPreviewMap[selectEl.value] || retargetPreviewMap["model.glb|Mixamo"];
        avatarPreviewImage.src = data.img;
        avatarPreviewName.textContent = data.name;
      }

      function mirrorValue(sourceId, targetId) {
        const source = document.getElementById(sourceId);
        const target = document.getElementById(targetId);
        if (!source || !target) return;
        target.value = source.value;
        target.dispatchEvent(new Event("change", { bubbles: true }));
      }

      function syncRetargetInputByExercise() {
  const exercise = document.getElementById("retargetExerciseSelect")?.value;
  const videoSelect = document.getElementById("retargetVideoSelect");
  const guideText = document.getElementById("retargetInputGuideText");
  const webcamSettings = document.getElementById("retargetWebcamSettings");
  const webcamModeSettings = document.getElementById("retargetWebcamModeSettings");

  if (!videoSelect) return;

  const isWebcam = exercise === "free";

  webcamSettings?.classList.toggle("hidden", !isWebcam);
  webcamModeSettings?.classList.toggle("hidden", !isWebcam);

  if (exercise === "free") {
    videoSelect.value = "";
    if (guideText) guideText.textContent = "자유 측정은 웹캠 입력을 사용합니다. 오버레이와 피드백이 활성화됩니다.";
  }

  if (exercise === "squat") {
    videoSelect.value = "squat2.mp4";
    if (guideText) guideText.textContent = "스쿼트 영상은 확인용 영상입니다. 오버레이와 피드백은 표시하지 않습니다.";
  }

  if (exercise === "stretching") {
    videoSelect.value = "stretching.mp4";
    if (guideText) guideText.textContent = "스트레칭 영상은 확인용 영상입니다. 오버레이와 피드백은 표시하지 않습니다.";
  }

  videoSelect.dispatchEvent(new Event("change", { bubbles: true }));
}

      function applyModeSettingsBeforeEnter(mode) {
        if (mode === "standard") {
  const standardVideo = document.getElementById("standardGuideVideoSelect");
  if (standardVideo) {
    standardVideo.value = "squat.mp4";
    standardVideo.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

        if (mode === "avatar") {
          mirrorValue("avatarModeAvatarSelect", "avatarSelect");
          mirrorValue("avatarModeGymSelect", "gymSelect");
        }

        if (mode === "retarget") {
          mirrorValue("retargetGymSelect", "gymSelect");
  syncRetargetInputByExercise();

  const exercise = document.getElementById("retargetExerciseSelect")?.value;
  const isVideoInput = exercise !== "free";

  window.HEALTH_MATE_RETARGET_VIDEO_ONLY = isVideoInput;
window.HEALTH_MATE_RETARGET_HIDE_VIDEO_OVERLAY = isVideoInput;
window.HEALTH_MATE_RETARGET_DISABLE_FEEDBACK = isVideoInput;
window.HEALTH_MATE_RETARGET_HIDE_MARKERS = isVideoInput;

  const setSource = document.getElementById("retargetSetCountInput");
  const repSource = document.getElementById("retargetRepCountInput");
  const restSource = document.getElementById("retargetRestSecondsInput");
  const typeSource = document.getElementById("retargetWorkoutTypeSelect");

  const setTarget = document.getElementById("setCountInput");
  const repTarget = document.getElementById("repCountInput");
  const restTarget = document.getElementById("restSecondsInput");
  const typeTarget = document.getElementById("workoutTypeSelect");

  if (exercise === "free") {
    if (setSource && setTarget) setTarget.value = setSource.value;
    if (repSource && repTarget) repTarget.value = repSource.value;
    if (restSource && restTarget) restTarget.value = restSource.value;
    if (typeSource && typeTarget) typeTarget.value = typeSource.value;

    setTarget?.dispatchEvent(new Event("change", { bubbles: true }));
    repTarget?.dispatchEvent(new Event("change", { bubbles: true }));
    restTarget?.dispatchEvent(new Event("change", { bubbles: true }));
    typeTarget?.dispatchEvent(new Event("change", { bubbles: true }));
  }
}
      }

      async function enterMode(mode) {

  window.showLoadingAvatar?.();

  await new Promise(r => setTimeout(r, 50)); // 렌더링 보장

  selectedMode = mode;

  applyModeSettingsBeforeEnter(mode);

  const avatarRuntimePanel = document.getElementById("avatarRuntimeSettings");
const retargetRuntimePanel = document.getElementById("retargetRuntimeSettings");

const showAvatarRuntimePanel =
  mode === "avatar" || mode === "standard";

avatarRuntimePanel?.classList.toggle("hidden", !showAvatarRuntimePanel);
retargetRuntimePanel?.classList.toggle("hidden", mode !== "retarget");

if (mode === "standard" || mode === "avatar") {
  const avatarValue = document.getElementById("avatarSelect")?.value || "default";
  const gymValue = document.getElementById("gymSelect")?.value || "gym1";

  const runtimeAvatar = document.getElementById("runtimeAvatarSelect");
  const runtimeGym = document.getElementById("runtimeGymSelect");

  if (runtimeAvatar) runtimeAvatar.value = avatarValue;
  if (runtimeGym) runtimeGym.value = gymValue;

  const avatarData = avatarPreviewMap[avatarValue] || avatarPreviewMap.default;
  const gymData = gymPreviewMap[gymValue] || gymPreviewMap.gym1;

  document.getElementById("runtimePreviewAvatarImg").src = avatarData.img;
  document.getElementById("runtimePreviewAvatarName").textContent = avatarData.name;
  document.getElementById("runtimePreviewGymImg").src = gymData.img;
  document.getElementById("runtimePreviewGymName").textContent = gymData.name;
}

document.getElementById("runtimePreviewPanel")?.classList.add("hidden");

  const topbar = document.querySelector(".app-topbar");
  const exercise = document.getElementById("retargetExerciseSelect")?.value;

  const hideRecordButtons =
  mode === "standard" ||
  (mode === "retarget" && exercise !== "free");

[
  "saveRecordBtn",
  "openRecordsBtn",
  "openAnalyticsBtn",
  "openRecordingsBtn",
  "userInfoPanel",
].forEach((id) => {
  document.getElementById(id)?.classList.toggle("hidden", hideRecordButtons);
});

const hideTopbar = false;
  if (topbar) {
    topbar.style.display = hideTopbar ? "none" : "flex";
  }

  const proxy = document.querySelector(`.mode-proxy-container .mode-card[data-mode="${mode}"]`);
  if (proxy) proxy.click();
    setTimeout(() => {
    window.hideLoadingAvatar?.();
  }, 1800);
}

function resetUiSettingsToDefault() {
  const defaults = {
    avatar: "default",
    gym: "gym1",
    workoutType: "normal",
    set: "3",
    rep: "10",
    rest: "30",
    retargetAvatar: "model.glb|Mixamo",
    retargetInput: "push_up",
  };

  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };

  setValue("avatarSelect", defaults.avatar);
  setValue("avatarModeAvatarSelect", defaults.avatar);
  setValue("runtimeAvatarSelect", defaults.avatar);

  setValue("gymSelect", defaults.gym);
  setValue("avatarModeGymSelect", defaults.gym);
  setValue("retargetGymSelect", defaults.gym);
  setValue("runtimeGymSelect", defaults.gym);
  setValue("runtimeRetargetGymSelect", defaults.gym);

  setValue("workoutTypeSelect", defaults.workoutType);
  setValue("setupAvatarWorkoutType", defaults.workoutType);
  setValue("retargetWorkoutTypeSelect", defaults.workoutType);
  setValue("runtimeWorkoutType", defaults.workoutType);
  setValue("runtimeRetargetWorkoutType", defaults.workoutType);

  setValue("setCountInput", defaults.set);
  setValue("repCountInput", defaults.rep);
  setValue("restSecondsInput", defaults.rest);
  setValue("runtimeSet", defaults.set);
  setValue("runtimeRep", defaults.rep);
  setValue("runtimeRest", defaults.rest);
  setValue("retargetSetCountInput", defaults.set);
  setValue("retargetRepCountInput", defaults.rep);
  setValue("retargetRestSecondsInput", defaults.rest);
  setValue("runtimeRetargetSet", defaults.set);
  setValue("runtimeRetargetRep", defaults.rep);
  setValue("runtimeRetargetRest", defaults.rest);

  setValue("retargetAvatarSelect", defaults.retargetAvatar);
  setValue("runtimeRetargetAvatarSelect", defaults.retargetAvatar);
  setValue("retargetExerciseSelect", defaults.retargetInput);
  setValue("runtimeRetargetInputSelect", defaults.retargetInput);

  window.HEALTH_MATE_RETARGET_VIDEO_ONLY = true;
  window.HEALTH_MATE_RETARGET_HIDE_VIDEO_OVERLAY = true;
  window.HEALTH_MATE_RETARGET_DISABLE_FEEDBACK = true;
  window.HEALTH_MATE_RETARGET_HIDE_MARKERS = true;

    document.getElementById("runtimeRetargetFreeSettings")?.classList.add("hidden");

  // ✅ 내부 상태까지 강제 기본값 동기화
  window.setWorkoutType?.("normal");
  window.setSetCount?.(3);
  window.setRepCount?.(10);
  window.setRestTime?.(30);
  window.changeRetargetInput?.("push_up");
}
window.resetUiSettingsToDefault = resetUiSettingsToDefault;

      startWorkoutBtn?.addEventListener("click", () => showPage("mode"));
      modeBackBtn?.addEventListener("click", () => showPage("home"));
      setupBackBtn?.addEventListener("click", () => {
   resetUiSettingsToDefault();
   showPage("mode");
 });

      const modeGrid = document.querySelector(".mode-grid");

modeGrid?.addEventListener("click", (e) => {
  const card = e.target.closest("[data-select-mode]");
  if (!card) return;

  const mode = card.dataset.selectMode;
  console.log("MODE CLICK:", mode);

  if (mode === "avatar") {
    enterMode("avatar");
    return;
  }

  if (mode === "retarget") {
  const retargetExerciseSelect = document.getElementById("retargetExerciseSelect");
  const runtimeInputSelect = document.getElementById("runtimeRetargetInputSelect");

  if (retargetExerciseSelect) {
    retargetExerciseSelect.value = "push_up";
    retargetExerciseSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }

  if (runtimeInputSelect) {
    runtimeInputSelect.value = "push_up";
    runtimeInputSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }

  enterMode("retarget");
  return;
}

  showSetupForMode(mode);
});

      document.querySelectorAll("[data-enter-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
          enterMode(btn.dataset.enterMode);
        });
      });

      document.getElementById("runtimeAvatarSelect")?.addEventListener("change", async (e) => {
  window.showLoadingAvatar?.();
  await new Promise(r => setTimeout(r, 50));

  window.setAvatar?.(e.target.value);

  setTimeout(() => window.hideLoadingAvatar?.(), 600);
});

document.getElementById("runtimeGymSelect")?.addEventListener("change", async (e) => {
  window.showLoadingAvatar?.();
  await new Promise(r => setTimeout(r, 50));

  window.setGym?.(e.target.value);

  setTimeout(() => window.hideLoadingAvatar?.(), 600);
});

document.getElementById("runtimeWorkoutType")?.addEventListener("change", (e) => {
  window.setWorkoutType?.(e.target.value);
});

document.getElementById("runtimeRetargetWorkoutType")?.addEventListener("change", (e) => {
  window.setWorkoutType?.(e.target.value);

  const isChallenge = e.target.value === "challenge";
  document.getElementById("runtimeRetargetFreeSettings")
    ?.classList.toggle("hidden", isChallenge);
});

document.getElementById("runtimeSet")?.addEventListener("change", (e) => {
  window.setSetCount?.(Number(e.target.value));
});

document.getElementById("runtimeRep")?.addEventListener("change", (e) => {
  window.setRepCount?.(Number(e.target.value));
});

document.getElementById("runtimeRest")?.addEventListener("change", (e) => {
  window.setRestTime?.(Number(e.target.value));
});

document.getElementById("runtimeRetargetInputSelect")?.addEventListener("change", (e) => {
  const input = e.target.value;
  const isFree = input === "free";

  document.getElementById("runtimeRetargetFreeSettings")
    ?.classList.toggle("hidden", !isFree);

  const setupSelect = document.getElementById("retargetExerciseSelect");
  if (setupSelect) {
    setupSelect.value = input;
    setupSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }

  window.changeRetargetInput?.(input);
});

      document.getElementById("studioStartBtn")?.addEventListener("click", () => {
  if (selectedMode === "avatar") {
    enterMode("avatar");
  }

  const isRetargetRuntime =
  selectedMode === "retarget" ||
  !document.getElementById("retargetRuntimeSettings")?.classList.contains("hidden");

if (isRetargetRuntime) {
    enterMode("retarget");
  }
});

      document.getElementById("avatarSelect")?.addEventListener("change", (e) => {
        if (selectedMode === "standard") syncAvatarPreviewFromSelect(e.target);
      });

      document.getElementById("gymSelect")?.addEventListener("change", (e) => {
        if (selectedMode === "standard") syncGymPreviewFromSelect(e.target);
      });

      document.getElementById("avatarModeAvatarSelect")?.addEventListener("change", (e) => {
        if (selectedMode === "avatar") syncAvatarPreviewFromSelect(e.target);
      });

      document.getElementById("avatarModeGymSelect")?.addEventListener("change", (e) => {
        if (selectedMode === "avatar") syncGymPreviewFromSelect(e.target);
      });

      document.getElementById("retargetAvatarSelect")?.addEventListener("change", () => {
        if (selectedMode === "retarget") updateRetargetPreview();
      });

      document.getElementById("retargetExerciseSelect")?.addEventListener("change", () => {
        if (selectedMode === "retarget") syncRetargetInputByExercise();
      });

      document.getElementById("retargetGymSelect")?.addEventListener("change", (e) => {
  if (selectedMode === "retarget") syncGymPreviewFromSelect(e.target);
});

      const workoutTypeSelect = document.getElementById("workoutTypeSelect");
      const normalWorkoutSettings = document.getElementById("normalWorkoutSettings");
      const challengeWorkoutSettings = document.getElementById("challengeWorkoutSettings");

      workoutTypeSelect?.addEventListener("change", () => {
        const isChallenge = workoutTypeSelect.value === "challenge";
        normalWorkoutSettings?.classList.toggle("hidden", isChallenge);
        challengeWorkoutSettings?.classList.toggle("hidden", !isChallenge);
      });

      const retargetWorkoutTypeSelect = document.getElementById("retargetWorkoutTypeSelect");
      const retargetChallengeInfo = document.getElementById("retargetChallengeInfo");

      retargetWorkoutTypeSelect?.addEventListener("change", () => {
        const isChallenge = retargetWorkoutTypeSelect.value === "challenge";
        retargetChallengeInfo?.classList.toggle("hidden", !isChallenge);
      });

      const bridgeClick = (fromId, toId) => {
        const from = document.getElementById(fromId);
        const to = document.getElementById(toId);
        from?.addEventListener("click", () => to?.click());
      };

      bridgeClick("modeRecordsBtn", "openRecordsBtn");
      bridgeClick("modeAnalyticsBtn", "openAnalyticsBtn");

      document.getElementById("pipSettingsBtn")?.addEventListener("click", () => {
  setTimeout(() => {
    const settingsPanel = document.getElementById("pipSettingsPanel");
    const previewPanel = document.getElementById("runtimePreviewPanel");

    const isOpen = settingsPanel && !settingsPanel.classList.contains("hidden");

    previewPanel?.classList.toggle("hidden", !isOpen);

    if (isOpen) updateRuntimePreviewPanel();
  }, 0);
});

[
  "runtimeAvatarSelect",
  "runtimeGymSelect",
  "runtimeRetargetAvatarSelect",
  "runtimeRetargetGymSelect",
].forEach((id) => {
  document.getElementById(id)?.addEventListener("change", () => {
    updateRuntimePreviewPanel();
  });
});

      showPage("home");
    })();