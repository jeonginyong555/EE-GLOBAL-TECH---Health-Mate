const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect("mongodb://127.0.0.1:27017/healthmate")
  .then(() => console.log("✅ MongoDB 연결 성공"))
  .catch(err => console.error("❌ DB 연결 에러:", err));

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

const Workout = mongoose.model("Workout", workoutSchema);

// public 폴더 기준 정적 서빙
const publicPath = path.resolve(__dirname, "..", "public");
app.use(express.static(publicPath));

app.get("/", (req, res) => {
  res.sendFile(path.join(publicPath, "index.html"));
});

app.post("/api/workout", async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(500).json({ message: "MongoDB not connected" });
    }

    const record = new Workout(req.body);
    await record.save();
    res.status(200).json({ message: "Success" });
  } catch (err) {
    console.error("❌ SAVE ERROR:", err);
    res.status(500).json({ message: "Save failed", error: err.message });
  }
});

app.get("/api/workout", async (req, res) => {
  try {
    const records = await Workout.find().sort({ timestamp: -1 });
    res.status(200).json(records);
  } catch (err) {
    console.error("❌ LOAD ERROR:", err);
    res.status(500).json({ message: "Load failed", error: err.message });
  }
});

app.get("/api/workout/:studentId", async (req, res) => {
  try {
    const records = await Workout.find({
      studentId: req.params.studentId
    }).sort({ timestamp: -1 });

    res.status(200).json(records);
  } catch (err) {
    console.error("❌ LOAD BY ID ERROR:", err);
    res.status(500).json({ message: "Load by id failed", error: err.message });
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
    res.status(500).json({ message: "Delete all failed", error: err.message });
  }
});

app.delete("/api/workout/record/:id", async (req, res) => {
  try {
    const deleted = await Workout.findByIdAndDelete(req.params.id);

    if (!deleted) {
      return res.status(404).json({ message: "해당 기록을 찾지 못함" });
    }

    res.status(200).json({
      message: "삭제 완료",
      deletedId: req.params.id
    });
  } catch (err) {
    console.error("❌ DELETE ONE ERROR:", err);
    res.status(500).json({ message: "Delete failed", error: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`🚀 서버 실행: http://localhost:${PORT}`));
