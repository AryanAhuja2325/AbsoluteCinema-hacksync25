const mongoose = require('mongoose');

const connect = async function () {
    mongoose.connect(process.env.MONGO_URI)
}

module.exports = { connect }