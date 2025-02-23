const { User } = require('../models/userModel');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

const signup = async (req, res) => {
    try {
        const data = req.body;
        const ALLOWED = ['firstName', 'lastName', 'email', 'password', 'phone', 'age', 'gender'];

        const isAllowed = Object.keys(data).every((key) => ALLOWED.includes(key));
        if (!isAllowed) {
            return res.status(400).json({
                success: false,
                message: "Invalid request"
            });
        }

        // Check if user with email already exists
        const existingUser = await User.findOne({ email: data.email });
        if (existingUser) {
            return res.status(409).json({
                success: false,
                message: "Email already in use. Please use a different email."
            });
        }

        const hashedPassword = await bcrypt.hash(data.password, 10);
        const user = new User({ ...data, password: hashedPassword });
        await user.save();

        res.status(200).json({
            success: true,
            message: "User created successfully",
            user: user
        });

    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        });
    }
};


const login = async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await User.findOne({ email });
        if (!user) {
            res.status(400).json({
                success: false,
                message: "Invalid Credentials"
            });
        }
        else {
            const check = await bcrypt.compare(password, user.password);
            if (check) {
                const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1d' });
                res.cookie('token', token, {
                    httpOnly: true,

                })
                res.status(200).json({
                    success: true,
                    message: "User logged in successfully",
                    user: user,
                    token: token
                })
            } else {
                res.status(400).json({
                    success: false,
                    message: "Invalid Credentials"
                })
            }
        }
    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        })
    }
}

const profile = async (req, res) => {
    try {
        const user = req.user;
        if (user)
            res.status(200).json({
                success: true,
                message: "User profile",
                user: user
            })
        else
            throw new Error("Please login first");
    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        })
    }
}

const updateUser = async (req, res) => {
    try {
        const id = req.user.id;
        const ALLOWED_UPDATES = ['firstName', 'lastName', 'password'];
        const data = req.body;
        const isAllowed = Object.keys(data).every(key => ALLOWED_UPDATES.includes(key));

        if (!isAllowed) {
            throw new Error("Invalid update");
        }
        if (data.password) {
            data.password = await bcrypt.hash(data.password, 10);
        }
        const user = await User.findByIdAndUpdate(id, data, { returnDocument: 'after', runValidators: true });
        req.user = user;
        res.status(200).json({
            success: true,
            message: "User updated successfully",
            user: req.user
        });
    } catch (err) {
        res.status(400).json({
            success: false,
            message: err.message
        })
    }
}
module.exports = { signup, login, profile, updateUser };