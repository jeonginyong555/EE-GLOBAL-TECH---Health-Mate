import express from 'express';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootPath = path.join(__dirname, '..');
const publicPath = path.join(rootPath, 'public');

app.use(express.json());

// public과 src 폴더를 모두 정적 폴더로 지정 (yoloWorker.js 호출용)
app.use(express.static(publicPath));
app.use('/src', express.static(__dirname)); 

// 1. 인용 님의 몽고DB 연결
mongoose.connect('mongodb://localhost:27017/healthmate')
    .then(() => console.log('✅ 인용님 몽고DB 연결 성공 (Path Fix 완료)'))
    .catch(err => console.error('❌ DB 연결 실패:', err));

const Workout = mongoose.model('Workout', new mongoose.Schema({
    studentId: String, count: Number, kst_time: String, mode: String, timestamp: { type: Date, default: Date.now }
}));

// 2. API 설정
app.post('/api/workout', async (req, res) => {
    try {
        const record = new Workout(req.body);
        await record.save();
        res.status(200).send({ ok: true });
    } catch (e) { res.status(500).send(e); }
});

// 3. [에러 해결!] 최신 Express 환경에서는 :splat* 형식을 사용해야 합니다.
app.get('/:splat*', (req, res) => {
    res.sendFile(path.join(publicPath, 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`🚀 서버 실행 중: http://localhost:3000`);
});