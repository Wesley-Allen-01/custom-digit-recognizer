#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <iostream>
#include "../include/cpu_matrix_ops.h"

class NeuralNet {
    private:
        int inputSize;
        int outputSize;
        std::vector<int> hiddenLayerSizes;
        std::vector<Matrix> weights;
        std::vector<Matrix> biases;
        std::vector<Matrix> activations;

    public:
        NeuralNet(int inputSize, std::vector<int> hiddenLayerSizes, int outputSize);
        void forward(const Matrix &input);
        void backward(const Matrix &target, float learningRate);
        void train(const Matrix &trainingData, const Matrix &trainingLabels, 
                   int epochs, float learningRate);
        std::vector<double> predict(const Matrix &input);
};

#endif