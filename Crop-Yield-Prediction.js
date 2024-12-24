// Import necessary libraries
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');

const app = express();
app.use(bodyParser.json());

// Step 1: Load Dataset
function loadData(filePath) {
    return new Promise((resolve, reject) => {
        const data = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (row) => data.push(row))
            .on('end', () => resolve(data))
            .on('error', (error) => reject(error));
    });
}

// Step 2: Preprocess Data
function preprocessData(data) {
    // Handle missing values by filtering out incomplete rows
    data = data.filter(row => Object.values(row).every(value => value !== null && value !== ''));

    // Convert categorical columns to numeric (e.g., one-hot encoding)
    const cropTypes = Array.from(new Set(data.map(row => row['Crop_Type'])));
    data = data.map(row => {
        cropTypes.forEach((type, index) => {
            row[`Crop_Type_${type}`] = row['Crop_Type'] === type ? 1 : 0;
        });
        delete row['Crop_Type'];
        return row;
    });

    // Normalize numeric features
    const numericKeys = Object.keys(data[0]).filter(key => key !== 'Yield');
    numericKeys.forEach(key => {
        const values = data.map(row => parseFloat(row[key]));
        const min = Math.min(...values);
        const max = Math.max(...values);
        data = data.map(row => {
            row[key] = (parseFloat(row[key]) - min) / (max - min);
            return row;
        });
    });

    // Split features and target
    const X = data.map(row => Object.keys(row).filter(key => key !== 'Yield').map(key => parseFloat(row[key])));
    const y = data.map(row => parseFloat(row['Yield']));

    return { X, y };
}

// Step 3: Train the Model
async function trainModel(X, y) {
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [X[0].length] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    const xs = tf.tensor2d(X);
    const ys = tf.tensor2d(y, [y.length, 1]);

    await model.fit(xs, ys, { epochs: 50, batchSize: 32 });

    return model;
}

// Step 4: Build Web App
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/predict', async (req, res) => {
    const inputs = req.body.inputs;
    const prediction = model.predict(tf.tensor2d([inputs]));
    const yieldPrediction = (await prediction.data())[0];
    res.json({ predictedYield: yieldPrediction.toFixed(2) });
});

// Main Function
(async function main() {
    const data = await loadData('crop_data.csv');
    const { X, y } = preprocessData(data);
    const model = await trainModel(X, y);

    app.listen(3000, () => console.log('App running on http://localhost:3000'));
})();
