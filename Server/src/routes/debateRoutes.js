const { createDebate, deleteDebate, getAllDebatesByUser, updateDebate, getDebate } = require('../controllers/debateControllers');
const { userAuth } = require('../middlewares/auth');
const express = require('express');

const debateRouter = express.Router();

debateRouter.post('/create', userAuth, createDebate);
debateRouter.delete('/delete/:id', userAuth, deleteDebate);
debateRouter.get('/getAllDebates', userAuth, getAllDebatesByUser);
debateRouter.get('/getDebate/:id', userAuth, getDebate);
debateRouter.patch('/updateDebate/:id', userAuth, updateDebate);

module.exports = { debateRouter };