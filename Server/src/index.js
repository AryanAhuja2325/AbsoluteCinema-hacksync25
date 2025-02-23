const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const cp = require('cookie-parser');
const { connect } = require('./config/database');
const { userRouter } = require('./routes/userRoutes');
const { debateRouter } = require('./routes/debateRoutes');
const { Server } = require('socket.io');
const http = require('http');
const { Debate } = require('./models/debateModel');
const jwt = require('jsonwebtoken');
const axios = require('axios');

dotenv.config();

const app = express();
const server = http.createServer(app);


const io = new Server(server, {
    cors: {
        origin: ['http://localhost:5173', 'http://localhost:5174'],
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    },
    cookie: true
})

io.on("connection", (socket) => {
    const cookie = socket.handshake.headers.token;

    if (!cookie) {
        socket.disconnect();
        return;
    }

    const decodeMsg = jwt.verify(cookie, process.env.JWT_SECRET);
    if (!decodeMsg) {
        socket.disconnect();
        return;
    }

    id = decodeMsg.id;
    if (!id) {
        socket.disconnect();
        return;
    }
    socket.userId = id;


    socket.on("debateMessage", async ({ message }) => {
        try {
            const { debateId, msg, mood, stand, topic } = message;
            const debate = await Debate.findById(debateId);

            if (!debate || debate.user.toString() !== socket.userId) {
                throw new Error("Debate not found");
            }

            let tempStream = [...debate.stream];
            tempStream.push({ sender: "User", message: msg });

            io.to(debateId).emit("messages", tempStream);

            const aiResponse = await axios.post(`${process.env.AI_URL}chat`, {
                input: msg,
                session_id: debateId,
                mood: mood.join(",").toString(),
                side: stand,
                topic
            });

            tempStream.push({ sender: "AI", message: aiResponse.data.answer });

            debate.stream = tempStream;
            await debate.save();

            io.to(debateId).emit("messages", debate.stream);
        } catch (err) {
            console.log("Error:", err.message);
            io.to(socket.id).emit("error", err.message);
            return;
        }
    });

    socket.on('disconnect', () => {
    });
});

app.use(express.json());
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:5174'],
    credentials: true,
}));
app.use(cp());
app.use(function (req, res, next) {
    res.header('Content-Type', 'application/json;charset=UTF-8')
    res.header('Access-Control-Allow-Credentials', true)
    res.header(
        'Access-Control-Allow-Headers',
        'Origin, X-Requested-With, Content-Type, Accept'
    )
    next()
})

app.use('/api/auth', userRouter)
app.use('/api/debate', debateRouter)

app.get('/', (req, res) => {
    res.send('hello');
});

connect().then(() => {
    server.listen(process.env.PORT, () => {
        console.log(`Server is running on ${process.env.PORT}`)
    })
}).catch((err) => {
    console.log(err);
})