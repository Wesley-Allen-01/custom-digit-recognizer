#include "../include/neuralNet.h"
#include <random>

NeuralNet::NeuralNet(int inputSize, std::vector<int> hiddenLayerSizes, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->hiddenLayerSizes = hiddenLayerSizes;

    std::srand(std::time(0));

    int previousLayerSize = inputSize;

    for (int i = 0; i < hiddenLayerSizes.size(); i++) {
        int layerSize = hiddenLayerSizes[i];
        Matrix weightMatrix(layerSize, std::vector<float>(previousLayerSize, 0));

        for (int j = 0; j < layerSize; j++) {
            for (int k = 0; k < previousLayerSize; k++) {
                weightMatrix[j][k] = (float) std::rand() / RAND_MAX * sqrt(2.0 / previousLayerSize);
            }
        }

        this->weights.push_back(weightMatrix);

        Matrix biasVector(layerSize, std::vector<float>(1, 0));
        this->biases.push_back(biasVector);

        previousLayerSize = layerSize;
    }
}

void NeuralNet::forward(const Matrix &input) {
    this->activations.clear();
    this->activations.push_back(input);

    Matrix currentInput = input;

    // normalize input values to be between 0 and 1
    currentInput = applyActivation(currentInput, sigmoid);

    for (int i = 0; i < this->weights.size(); i++) {
        Matrix currentWeights = this->weights[i];
        Matrix currentBiases = this->biases[i];

        currentInput = matMul(currentWeights, currentInput);
        currentInput = matAdd(currentInput, currentBiases);

        if (i != this->weights.size() - 1) {
            currentInput = applyActivation(currentInput, sigmoid);
        } else {
            // currentInput = applyActivation(currentInput, softmax);
        }


        this->activations.push_back(currentInput);
    }

}


    
    
