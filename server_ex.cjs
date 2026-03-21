const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB 연결
mongoose.connect('mongodb://localhost:27017/healthmate')
    .then(() => console.log('✅ MongoDB 연결 성공'))
    .catch(err => console.error('❌ DB 연결 에러:', err));

// 스키마 정의
const workoutSchema = new mongoose.Schema({
    studentId: String,
    name: String,
    count: Number,
    timestamp: { type: Date, default: Date.now }
});
const Workout = mongoose.model('Workout', workoutSchema);

const publicPath = path.resolve(__dirname, '..', 'public');
app.use(express.static(publicPath));

// 데이터 저장 API
app.post('/api/workout', async (req, res) => {
    try {
        const record = new Workout(req.body);
        await record.save();
        res.status(200).send({ message: "Success" });
    } catch (err) {
        res.status(500).send(err);
    }
});

app.listen(3000, () => console.log(`🚀 서버 실행: http://localhost:3000`));