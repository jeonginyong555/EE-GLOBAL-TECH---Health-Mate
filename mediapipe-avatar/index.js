import express from 'express';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());

// 1. [핵심 수정] 정적 파일의 위치를 'public' 폴더로 지정
// 이제 서버는 모든 html, js, 모델 파일을 public 폴더 안에서 찾습니다.
app.use(express.static(path.join(__dirname, 'public')));

// 2. MongoDB 연결
mongoose.connect('mongodb://localhost:27017/healthmate')
    .then(() => console.log("✅ MongoDB 연결 성공!"))
    .catch(err => console.error("❌ MongoDB 연결 에러:", err));

const Squat = mongoose.model('Squat', new mongoose.Schema({
    user: { type: String, default: "Jeong In-yong" },
    count: Number,
    timestamp: { type: Date, default: Date.now }
}));

// 3. API 엔드포인트
app.get('/api/squat/history', async (req, res) => {
    try {
        const history = await Squat.find().sort({ timestamp: -1 }).limit(5);
        res.json(history);
    } catch (e) { res.status(500).json([]); }
});

app.post('/api/squat/save', async (req, res) => {
    try {
        await new Squat(req.body).save();
        res.status(200).send("Saved");
    } catch (e) { res.status(500).send(e); }
});

// 4. [수정] 루트 접속 시 public 폴더 안의 index.html을 보내줌
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = 8080;
app.listen(PORT, () => {
    console.log(`
    🚀 서버 실행 중: http://localhost:${PORT}
    📂 리소스 폴더: ${path.join(__dirname, 'public')}
    `);
});