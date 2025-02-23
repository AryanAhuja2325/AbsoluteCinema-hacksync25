const mongoose = require('mongoose');
const validator = require('validator');

const userSchema = new mongoose.Schema({
    firstName: {
        type: String,
        required: true,
        trim: true
    },
    lastName: {
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true,
        immutable: true,
        validate: function (val) {
            if (!validator.isEmail(val)) {
                throw new Error("Invalid email id: " + val);
            }
        }
    },
    password: {
        type: String,
        required: true,
        validate: function (val) {
            if (!validator.isStrongPassword(val)) {
                throw new Error("Password is not strong enough");
            }
        }
    },
}, { timestamps: true });

const User = mongoose.model('user', userSchema);

module.exports = { User };