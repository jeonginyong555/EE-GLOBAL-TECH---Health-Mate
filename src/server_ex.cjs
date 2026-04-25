const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const multer = require("multer");

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true }));

mongoose.connect("mongodb://127.0.0.1:27017/healthmate")
  .then(() => console.log("✅ MongoDB 연결 성공"))
  .catch(err => console.error("❌ DB 연결 에러:", err));

const publicPath = path.resolve(__dirname, "..", "public");
const uploadPath = path.join(publicPath, "uploads");

if (!fs.existsSync(uploadPath)) {
  fs.mkdirSync(uploadPath, { recursive: true });
  console.log("📁 uploads 폴더 생성:", uploadPath);
}

app.use(express.static(publicPath));
app.use("/uploads", express.static(uploadPath));

/* =========================
 * SCHEMA
 * ========================= */
const workoutSchema = new mongoose.Schema({
  studentId: String,
  name: String,
  count: Number,
  avgDepth: Number,
  depthLowCount: Number,
  torsoWarningCount: Number,
  mode: String,
  inputMode: String,
  setResults: Array,
  kst_time: String,
  timestamp: { type: Date, default: Date.now }
});

const recordingSchema = new mongoose.Schema({
  studentId: String,
  name: String,
  mode: String,
  filename: String,
  url: String,
  mimetype: String,
  size: Number,
  kst_time: String,
  timestamp: { type: Date, default: Date.now }
});

const Workout = mongoose.model("Workout", workoutSchema);
const Recording = mongoose.model("Recording", recordingSchema);

/* =========================
 * MULTER
 * ========================= */
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
  let ext = path.extname(file.originalname || "");

  if (!ext) {
    if (file.mimetype === "video/webm") ext = ".webm";
    else if (file.mimetype === "video/mp4") ext = ".mp4";
    else ext = ".webm";
  }

  cb(null, `recording_${Date.now()}${ext}`);
}
});

const upload = multer({
  storage,
  limits: {
    fileSize: 200 * 1024 * 1024
  }
});

/* =========================
 * PAGE ROUTES
 * ========================= */
app.get("/", (req, res) => {
  res.sendFile(path.join(publicPath, "index.html"));
});

app.get("/records", (req, res) => {
  res.sendFile(path.join(publicPath, "records.html"));
});

app.get("/analytics", (req, res) => {
  res.sendFile(path.join(publicPath, "analytics.html"));
});

/* =========================
 * WORKOUT API
 * ========================= */
app.post("/api/workout", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const record = new Workout(req.body);
    await record.save();

    res.status(200).json({ message: "Success", savedId: record._id });
  } catch (err) {
    console.error("❌ SAVE ERROR:", err);
    res.status(500).json({
      message: "Save failed",
      error: err.message
    });
  }
});

app.get("/api/workout", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const records = await Workout.find().sort({ timestamp: -1 });
    res.status(200).json(records);
  } catch (err) {
    console.error("❌ LOAD ERROR:", err);
    res.status(500).json({
      message: "Load failed",
      error: err.message
    });
  }
});

app.get("/api/workout/:studentId", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const records = await Workout.find({
      studentId: req.params.studentId
    }).sort({ timestamp: -1 });

    res.status(200).json(records);
  } catch (err) {
    console.error("❌ LOAD BY ID ERROR:", err);
    res.status(500).json({
      message: "Load by id failed",
      error: err.message
    });
  }
});

app.delete("/api/workout", async (req, res) => {
  try {
    const result = await Workout.deleteMany({});
    res.status(200).json({
      message: "전체 삭제 완료",
      deletedCount: result.deletedCount
    });
  } catch (err) {
    console.error("❌ DELETE ALL ERROR:", err);
    res.status(500).json({
      message: "Delete all failed",
      error: err.message
    });
  }
});

app.delete("/api/workout/record/:id", async (req, res) => {
  try {
    const deleted = await Workout.findByIdAndDelete(req.params.id);

    if (!deleted) {
      return res.status(404).json({
        message: "해당 기록을 찾지 못함"
      });
    }

    res.status(200).json({
      message: "삭제 완료",
      deletedId: req.params.id
    });
  } catch (err) {
    console.error("❌ DELETE ONE ERROR:", err);
    res.status(500).json({
      message: "Delete failed",
      error: err.message
    });
  }
});

/* =========================
 * RECORDING API
 * ========================= */
app.post("/api/recording", upload.single("video"), async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    if (!req.file) {
      return res.status(400).json({ message: "video file is required" });
    }

    const record = new Recording({
      studentId: req.body.studentId || "",
      name: req.body.name || "",
      mode: req.body.mode || "avatar",
      filename: req.file.filename,
      url: `/uploads/${req.file.filename}`,
      mimetype: req.file.mimetype,
      size: req.file.size,
      kst_time: req.body.kst_time || new Date().toLocaleString("ko-KR", {
        timeZone: "Asia/Seoul"
      })
    });

    await record.save();

    res.status(200).json({
      message: "Recording saved",
      savedId: record._id,
      url: record.url
    });
  } catch (err) {
    console.error("❌ RECORDING SAVE ERROR:", err);
    res.status(500).json({
      message: "Recording save failed",
      error: err.message
    });
  }
});

app.get("/api/recording", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const records = await Recording.find().sort({ timestamp: -1 });
    res.status(200).json(records);
  } catch (err) {
    console.error("❌ RECORDING LOAD ERROR:", err);
    res.status(500).json({
      message: "Recording load failed",
      error: err.message
    });
  }
});

app.get("/api/recording/:studentId", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const records = await Recording.find({
      studentId: req.params.studentId
    }).sort({ timestamp: -1 });

    res.status(200).json(records);
  } catch (err) {
    console.error("❌ RECORDING LOAD BY ID ERROR:", err);
    res.status(500).json({
      message: "Recording load by id failed",
      error: err.message
    });
  }
});

app.delete("/api/recording/record/:id", async (req, res) => {
  try {
    const deleted = await Recording.findByIdAndDelete(req.params.id);

    if (!deleted) {
      return res.status(404).json({
        message: "해당 녹화 기록을 찾지 못함"
      });
    }

    if (deleted.filename) {
      const filePath = path.join(uploadPath, deleted.filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }

    res.status(200).json({
      message: "녹화 삭제 완료",
      deletedId: req.params.id
    });
  } catch (err) {
    console.error("❌ RECORDING DELETE ERROR:", err);
    res.status(500).json({
      message: "Recording delete failed",
      error: err.message
    });
  }
});

/* =========================
 * SERVER
 * ========================= */
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 서버 실행: http://localhost:${PORT}`);
  console.log(`📁 정적 경로: ${publicPath}`);
  console.log(`🎥 업로드 경로: ${uploadPath}`);
});