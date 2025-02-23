const jwt = require('jsonwebtoken');
const { User } = require('../models/userModel');

const userAuth = async (req, res, next) => {
    try {
        const { token } = req.headers;
        // console.log(req.headers);
        
        if (!token) {
            throw new Error("Please login first");
        }
        const decodedMsg = jwt.verify(token, process.env.JWT_SECRET);
        const user = await User.findById(decodedMsg.id);

        if (!user) {
            throw new Error("Please login first");
        }
        req.user = user;
        next();
    } catch (err) {
        return res.status(401).json({
            success: false,
            message: err.message
        })
    }
}
module.exports = { userAuth };