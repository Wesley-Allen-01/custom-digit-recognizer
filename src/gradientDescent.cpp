#include "../include/neuralNet.h"
#include "../include/cpu_matrix_ops.h"
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;

/*
Steps for Implementing Gradient Descent

1. Load Data
2. Define Loss Function
3. Training Loop for Batches
4. Mini-Batch Training (time-permitting)
5. Test Data

*/


/*
LOAD DATA
*/

pair<vector<Matrix>, vector<Matrix>> loadMNISTData(const string &filename) {
    vector<Matrix> data;
    vector<Matrix> labels;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string header;
    getline(file, header);  // discard the header line

    while (getline(file, line)) {
        stringstream ss(line);
        string value;

        // Read label (first column)
        getline(ss, value, ',');
        int label = stoi(value);
        Matrix labelMatrix(1, vector<float>(10, 0.0f));
        labelMatrix[0][label] = 1.0f; // One-hot encoding for the label
        labels.push_back(labelMatrix);

        // Read pixel data
        Matrix matrix(1, vector<float>(0));  
        while (getline(ss, value, ',')) {
            matrix[0].push_back(stof(value) / 255.0f); //normalize 
        }
        data.push_back(matrix);
    }
    file.close();
    return make_pair(data, labels);
}

/*
LOSS FUNCTION (using MSE, but could try different losss functions and evaluate)
*/


int argMax(const Matrix &vec) {
    int maxIdx = 0;
    float maxVal = vec[0][0];
    for(int j = 0; j < vec[0].size(); j++){
        if(vec[0][j] > maxVal){
            maxVal = vec[0][j];
            maxIdx = j;
        }
    }
    return maxIdx;
}

float computeAccuracy(const vector<Matrix> &predictions,
                      const vector<Matrix> &labels) {
    int correct = 0;
    for(int i = 0; i < predictions.size(); i++){
        if(argMax(predictions[i]) == argMax(labels[i])){
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size();
}

float avgMSE(const Matrix &output, const Matrix &target){
    if (output.empty() || target.empty()) {
        cerr << "Error: One of the matrices is empty in avgMSE." << endl;
        exit(EXIT_FAILURE);
    }
    float sum = 0.0f;
    int rows = output.size();
    int cols = output[0].size();

     if (target.size() != rows || target[0].size() != cols) {
        cerr << "Error: Matrix size mismatch in avgMSE. Output has size: " 
             << rows << "x" << cols 
             << ", Target has size: " << target.size() 
             << "x" << (target.empty() ? 0 : target[0].size()) << endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            float diff = output[i][j] - target[i][j];
            sum += diff*diff;
        }

    }

    float avg_MSE = sum/(rows*cols);
    return avg_MSE;
}


/*
TRAINING 
*/


int main() {
    int passes = 5;
    float learningRate = 0.01f;
    float accuracyThreshold = 0.9f;

    auto [trainingData, trainingLabels] = loadMNISTData("data/mnist_train.csv");

    //debug

    NeuralNet net = NeuralNet(784, {128, 64}, 10);

    for(int pass = 0; pass < passes; pass++){
        float passLoss = 0.0f;
        vector<Matrix> passPredictions;
        for (int i = 0; i < trainingData.size();i++){
            net.forward(trainingData[i]);

            Matrix output = net.getActivations().back();
   

            Matrix transposedOutput = transpose(output);
            float loss = avgMSE(transposedOutput,trainingLabels[i]);
            passLoss += loss;
  

            net.backward(trainingLabels[i],learningRate);
      
            passPredictions.push_back(output);
        }

        passLoss /= trainingData.size();
        

        float accuracy = computeAccuracy(passPredictions, trainingLabels);
        
        cout << "Pass " << pass << " - Loss: " << passLoss << " - Accuracy: " << accuracy << endl; 

        if(accuracy >= accuracyThreshold){
            cout << "Reached desired accuracy of " << accuracy << " on pass " << pass << ". Stopping training." << endl;
            break;
        }
    } 
}