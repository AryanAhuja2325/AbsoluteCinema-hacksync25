const mongoose = require('mongoose');
const validators = require('validator');

const debateSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
        trim: true,
        minLength: 3,
        maxLength: 100
    },
    mood: {
        type: [String],
        required: true,
        enum: ['Passionate', 'Aggressive', 'Defensive', 'Skeptical', 'Thoughtful', 'Collaborative', 'Playful']
    },
    topic: {
        type: String,
        required: true,
        trim: true,
        minLength: 3,
        maxLength: 200
    },
    aiInclination: {
        type: String,
        required: true,
        enum: ['For', 'Against', 'Neutral']
    },
    stream: [
        {
            sender: {
                type: String,
                required: true,
                enum: ['User', 'AI']
            },
            message: {
                type: String,
                required: true,
                trim: true,
                minLength: 1,
                maxLength: 1000
            },
            timestamp: {
                type: Date,
                default: Date.now
            }
        }
    ],
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }
}, { timestamps: true });


const Debate = mongoose.model('Debate', debateSchema);
module.exports = { Debate };