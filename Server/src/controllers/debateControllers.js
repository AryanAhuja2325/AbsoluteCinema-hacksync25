const { Debate } = require('../models/debateModel');

const createDebate = async (req, res, next) => {
    try {
        const { mood, topic, aiInclination } = req.body;
        const user = req.user;

        const aiObj = {
            sender: "AI",
            message: "Welcome to DebateAI API. Let's start the debate",
        }

        const debate = new Debate({
            mood,
            topic,
            aiInclination,
            user: user._id,
            title: topic,
            stream: [
                aiObj
            ]
        });
        await debate.save();

        res.status(200).json({
            success: true,
            message: "Debate created successfully",
            debate
        })
    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        })
    }
}

const deleteDebate = async (req, res, next) => {
    try {
        const { id } = req.params;
        const debate = await Debate.findById(id);

        if (debate.user.toString() !== req.user._id.toString()) {
            throw new Error("You are not authorized to delete this debate");
        }

        if (!debate) {
            throw new Error("Debate not found");
        }
        await debate.deleteOne();
        res.status(200).json({
            success: true,
            message: "Debate deleted successfully"
        })
    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        })
    }
}

const getAllDebatesByUser = async (req, res) => {
    try {
        const user = req.user;
        const debates = await Debate.find({ user: user._id });
        res.status(200).json({
            success: true,
            message: "Debates fetched successfully",
            debates
        })
    } catch (error) {
        res.status(400).json({
            success: false,
            message: error.message
        })
    }
}

const getDebate = async (req, res) => {
    try {
        const { id } = req.params;
        const debate = await Debate.findById(id);

        if (!debate)
            throw new Error("Debate not found");

        res.status(200).json({
            success: true,
            message: "Debate fetched successfully",
            debate
        });
    } catch (err) {
        res.status(400).json({
            success: false,
            message: err.message
        })
    }
}

const updateDebate = async (req, res) => {
    try {
        const { id } = req.params;
        const ALLOWED_UPDATES = ['mood', 'topic', 'aiInclination', 'title'];
        const data = req.body;

        const isAllowed = Object.keys(data).every(key => ALLOWED_UPDATES.includes(key));
        if (!isAllowed) {
            throw new Error("Invalid update");
        }
        await Debate.findByIdAndUpdate(id, data, { new: true, runValidators: true });
        res.status(200).json({
            success: true,
            message: "Debate updated successfully",
            data
        });
    } catch (err) {
        res.status(400).json({
            success: false,
            message: err.message
        })
    }
}

module.exports = { createDebate, deleteDebate, getAllDebatesByUser, getDebate, updateDebate };