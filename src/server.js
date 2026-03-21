const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// 1. MongoDB 연결 (인용님 설정 유지)
mongoose.connect('mongodb://localhost:27017/healthmate')
    .then(() => console.log('✅ MongoDB 연결 성공'))
    .catch(err => console.error('❌ DB 연결 에러:', err));

const workoutSchema = new mongoose.Schema({
    studentId: String,
    name: String,
    count: Number,
    kst_time: String,
    timestamp: { type: Date, default: Date.now }
});
const Workout = mongoose.model('Workout', workoutSchema);

// 2. [핵심 수정] 정적 파일 제공 경로
// __dirname은 현재 파일(server.js)이 있는 src 폴더입니다.
// '../'를 통해 상위 폴더(루트)로 나가서 모든 파일을 찾게 합니다.
const rootPath = path.join(__dirname, '../'); 
app.use(express.static(rootPath));

// 3. API 경로
app.post('/api/workout', async (req, res) => {
    try {
        const record = new Workout(req.body);
        await record.save();
        res.status(200).send({ ok: true });
    } catch (err) {
        res.status(500).send({ ok: false, error: err.message });
    }
});

// 4. 루트 접속 시 index.html 파일을 직접 전송
app.get('/', (req, res) => {
    res.sendFile(path.join(rootPath, 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`
🚀 서버가 성공적으로 실행되었습니다!
🌍 접속 주소: http://localhost:${PORT}
📂 파일 경로: ${rootPath}
    `);
});