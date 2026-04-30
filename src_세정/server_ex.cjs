const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();

// 1. 미들웨어 설정
app.use(cors());
app.use(express.json()); // JSON 형태의 데이터를 받기 위함

// 2. MongoDB 연결 (데이터베이스 이름: healthmate)
mongoose.connect('mongodb://localhost:27017/healthmate')
    .then(() => console.log('✅ MongoDB 연결 성공'))
    .catch(err => console.error('❌ DB 연결 에러:', err));

// 3. 스키마 정의 (main2.js가 보내는 상세 데이터를 모두 담을 수 있게 확장)
const sessionSchema = new mongoose.Schema({
    session_name: String,
    exercise_type: String,      // squat 또는 pushup
    total_count: Number,
    avg_depth: Number,
    depth_low_count: Number,
    torso_warning_count: Number,
    duration_sec: Number,
    input_mode: String,
    mode: String,               // normal 또는 challenge
    set_results: Array,         // 세트별 상세 결과 (배열)
    created_at: { 
        type: String, 
        default: () => new Date().toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' }) 
    }
});

const Session = mongoose.model('Session', sessionSchema);

// 4. 정적 파일 경로 설정 (중요!)
// server_ex.cjs와 public 폴더가 같은 위치에 있다면 아래 코드가 맞습니다.
const publicPath = path.join(__dirname, 'public'); 
app.use(express.static(publicPath));

/** ===============================================
 * API 라우트 (main2.js와 연결되는 부분)
 * =============================================== */

// [POST] 운동 결과 저장
app.post('/api/sessions', async (req, res) => {
    try {
        // main2.js의 payload 필드명과 DB 필드명을 맞춥니다.
        const data = {
            session_name: req.body.sessionName,
            exercise_type: req.body.exerciseType,
            total_count: req.body.totalCount,
            avg_depth: req.body.avgDepth,
            depth_low_count: req.body.depthLowCount,
            torso_warning_count: req.body.torsoWarningCount,
            duration_sec: req.body.durationSec,
            input_mode: req.body.inputMode,
            mode: req.body.mode,
            set_results: req.body.setResults
        };

        const newSession = new Session(data);
        await newSession.save();
        res.status(200).json({ ok: true, message: "저장 성공" });
    } catch (err) {
        console.error("저장 에러:", err);
        res.status(500).json({ ok: false, message: err.message });
    }
});

// [GET] 전체 기록 목록 불러오기 (정렬 기능 포함)
app.get('/api/sessions', async (req, res) => {
    try {
        const { sort } = req.query;
        let sortOption = { _id: -1 }; // 기본: 최신순

        if (sort === 'oldest') sortOption = { _id: 1 };
        if (sort === 'count_desc') sortOption = { total_count: -1 };
        if (sort === 'depth_desc') sortOption = { avg_depth: -1 };

        const rows = await Session.find().sort(sortOption);
        res.status(200).json({ ok: true, rows });
    } catch (err) {
        res.status(500).json({ ok: false, message: err.message });
    }
});

// [GET] 가장 최근 기록 하나만 불러오기
app.get('/api/sessions/latest', async (req, res) => {
    try {
        const row = await Session.findOne().sort({ _id: -1 });
        res.status(200).json({ ok: true, row });
    } catch (err) {
        res.status(500).json({ ok: false, message: err.message });
    }
});

// [DELETE] 특정 기록 삭제
app.delete('/api/sessions/:id', async (req, res) => {
    try {
        await Session.findByIdAndDelete(req.params.id);
        res.status(200).json({ ok: true, message: "삭제 성공" });
    } catch (err) {
        res.status(500).json({ ok: false, message: err.message });
    }
});

// 5. 서버 시작 (포트 3000)
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`
🚀 Health-Mate 서버 실행 중!
🔗 접속 주소: http://localhost:${PORT}
📁 정적 경로: ${publicPath}
    `);
});