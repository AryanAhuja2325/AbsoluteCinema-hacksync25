const express = require('express');
const userRouter = express.Router();
const { signup, login, profile, updateUser } = require('../controllers/userControllers');
const { userAuth } = require('../middlewares/auth');

userRouter.post('/signup', signup)
userRouter.post('/login', login)
userRouter.get('/profile', userAuth, profile)
userRouter.patch('/updateUser', userAuth, updateUser);

module.exports = { userRouter };