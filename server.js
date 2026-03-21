import express from "express";
import cors from "cors";
import sqlite3 from "sqlite3";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 4000;

app.use(cors());
app.use(express.json());

const dbPath = path.join(__dirname, "healthmate.db");
const db = new sqlite3.Database(dbPath);

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS squat_sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_name TEXT,
      total_count INTEGER NOT NULL DEFAULT 0,
      avg_depth REAL NOT NULL DEFAULT 0,
      depth_low_count INTEGER NOT NULL DEFAULT 0,
      torso_warning_count INTEGER NOT NULL DEFAULT 0,
      duration_sec REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
  `);
});

app.post("/api/sessions", (req, res) => {
  const { sessionName, totalCount, avgDepth, depthLowCount, torsoWarningCount, durationSec } = req.body;
  db.run(
    `INSERT INTO squat_sessions (session_name, total_count, avg_depth, depth_low_count, torso_warning_count, duration_sec) VALUES (?, ?, ?, ?, ?, ?)`,
    [sessionName, totalCount, avgDepth, depthLowCount, torsoWarningCount, durationSec],
    function (err) {
      if (err) return res.status(500).json({ ok: false, message: err.message });
      res.json({ ok: true, id: this.lastID });
    }
  );
});

app.get("/api/sessions", (req, res) => {
  db.all(`SELECT * FROM squat_sessions ORDER BY id DESC`, [], (err, rows) => {
    if (err) return res.status(500).json({ ok: false });
    res.json({ ok: true, rows });
  });
});

app.listen(PORT, () => console.log(`DB Server running on http://localhost:${PORT}`));