#include "../include/neuralNet.h"
#include "../include/cpu_matrix_ops.h"
#include <random>
#include <iostream>
#include <vector>
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

//I don't have the MNIST set so can you implement Wes?
vector<Matrix> trainingData;
vector<Matrix> trainingLabels;

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
    float sum = 0.0f;
    int rows = output.size();
    int cols = output[0].size();
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
    int passes = 100;
    float learningRate = 0.1f;
    float accuracyThreshold = 0.9f;

    NeuralNet net = NeuralNet(784, {128, 64}, 10);

    for(int pass = 0; pass < passes; pass++){
        float passLoss = 0.0f;
        vector<Matrix> passPredictions;
        for (int i = 0; i < trainingData.size();i++){
            net.forward(trainingData[i]);
            Matrix output = net.getActivations().back();

            float loss = avgMSE(output,trainingLabels[i]);
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
