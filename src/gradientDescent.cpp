#include "../include/neuralNet.h"
#include "../include/cpu_matrix_ops.h"
#include <random>
#include <iostream>
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

    avgMSE = sum/(rows*cols);
    return avgMSE;
}


/*
TRAINING 
*/

int passes = 100;
float learningRate = 0.1f;
float accuracyThreshold = 0.9f;

for(int pass = 0; pass < passes; pass++){
    flaot passLoss = 0.0f
    vector<Matrix> passPredictions;
    for (int i = 0; i < trainingData.size();i++){
        //run the forward pass
        net.forward(trainingData[i]);
        Matrix output = net.getActivations().back();

        //calculate avgMSE
        float loss = avgMSE(output,trainingLabels[i]);
        passLoss += loss;

        //run back propagation and update weights
        net.backward(trainingLabels[i],learningRate);

        //store the predictions for accuracy
        passPredictions.push_back(output);
    }

    //print avg loss for debugging
    passLoss /= trainingData.size();
    cout << "Pass " << pass << " - Loss: " << passLoss << " - Accuracy: " << accuracy << endl; 

    float accuracy = computeAccuracy(passPredictions, trainingLabels);
    
    if(accuracy >= accuracyThreshold){
        cout << "Reached desired accuracy of " << accuracy << " on pass " << pass << ". Stopping training." << endl;
        break;
    }
}



