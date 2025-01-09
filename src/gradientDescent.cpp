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

for(int pass = 0; pass < passes; pass++){
    flaot passLoss = 0.0f

    for (int i = 0; i < trainingData.size();i++){
        //run the forward pass
        net.forward(trainingData[i]);
        Matrix output = net.getActivations().back();

        //calculate avgMSE
        float loss = avgMSE(output,trainingLabels[i]);
        passLoss += loss;

        //run back propagation and update weights
        net.backward(trainingLabels[i],learningRate);
    }

    //print avg loss for debugging
    passLoss /= trainingData.size();
    cout << "Pass " << pass << " - Loss: " << passLoss << endl; 

}



