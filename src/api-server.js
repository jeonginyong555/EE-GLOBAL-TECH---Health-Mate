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

app.get("/", (req, res) => {
  res.send("Health-Mate SQLite API Server is running.");
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true });
});

app.get("/api/sessions", (req, res) => {
  const sort = String(req.query.sort || "latest");

  let orderBy = "id DESC";
  if (sort === "oldest") orderBy = "id ASC";
  else if (sort === "count_desc") orderBy = "total_count DESC, id DESC";
  else if (sort === "depth_desc") orderBy = "avg_depth DESC, id DESC";

  db.all(
    `SELECT * FROM squat_sessions ORDER BY ${orderBy} LIMIT 50`,
    [],
    (err, rows) => {
      if (err) {
        return res.status(500).json({ ok: false, message: err.message });
      }
      res.json({ ok: true, rows });
    }
  );
});

app.get("/api/sessions/latest", (req, res) => {
  db.get(
    `SELECT * FROM squat_sessions ORDER BY id DESC LIMIT 1`,
    [],
    (err, row) => {
      if (err) {
        return res.status(500).json({ ok: false, message: err.message });
      }
      res.json({ ok: true, row: row || null });
    }
  );
});

app.post("/api/sessions", (req, res) => {
  const {
    sessionName,
    totalCount,
    avgDepth,
    depthLowCount,
    torsoWarningCount,
    durationSec,
  } = req.body || {};

  db.run(
    `
    INSERT INTO squat_sessions
    (session_name, total_count, avg_depth, depth_low_count, torso_warning_count, duration_sec)
    VALUES (?, ?, ?, ?, ?, ?)
    `,
    [
      sessionName ?? "squat_session",
      Number(totalCount ?? 0),
      Number(avgDepth ?? 0),
      Number(depthLowCount ?? 0),
      Number(torsoWarningCount ?? 0),
      Number(durationSec ?? 0),
    ],
    function (err) {
      if (err) {
        return res.status(500).json({ ok: false, message: err.message });
      }

      db.get(
        `SELECT * FROM squat_sessions WHERE id = ?`,
        [this.lastID],
        (err2, row) => {
          if (err2) {
            return res.status(500).json({ ok: false, message: err2.message });
          }
          res.json({ ok: true, row });
        }
      );
    }
  );
});

app.delete("/api/sessions/:id", (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isFinite(id)) {
    return res.status(400).json({ ok: false, message: "invalid id" });
  }

  db.run(`DELETE FROM squat_sessions WHERE id = ?`, [id], function (err) {
    if (err) {
      return res.status(500).json({ ok: false, message: err.message });
    }

    res.json({
      ok: true,
      deletedId: id,
      changes: this.changes || 0,
    });
  });
});

app.listen(PORT, () => {
  console.log(`SQLite API server running on http://localhost:${PORT}`);
});