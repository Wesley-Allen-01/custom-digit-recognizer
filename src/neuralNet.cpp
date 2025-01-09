#include "../include/neuralNet.h"
#include "../include/cpu_matrix_ops.h"
#include <random>
#include <iostream>
using namespace std; 
NeuralNet::NeuralNet(int inputSize, std::vector<int> hiddenLayerSizes, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->hiddenLayerSizes = hiddenLayerSizes;

    std::srand(std::time(0));

    int previousLayerSize = inputSize;

    // 1) Allocate weights/biases for each hidden layer
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

    // 2) Allocate weights/biases for the final (output) layer
    //    previousLayerSize is now size of last hidden layer
    Matrix outputWeightMatrix(outputSize, std::vector<float>(previousLayerSize, 0));
    for(int r = 0; r < outputSize; r++){
        for(int c = 0; c < previousLayerSize; c++){
            outputWeightMatrix[r][c] = (float) std::rand() / RAND_MAX;
        }
    }
    this->weights.push_back(outputWeightMatrix);

    // bias is [outputSize x 1]
    Matrix outputBiasVector(outputSize, std::vector<float>(1, 0));
    this->biases.push_back(outputBiasVector);
}

void NeuralNet::forward(const Matrix &input) {
    this->activations.clear();
    this->activations.push_back(input);

    Matrix currentInput = input;

    for (int i = 0; i < this->weights.size(); i++) {
        Matrix currentWeights = this->weights[i];
        Matrix currentBiases = this->biases[i];

        currentInput = matMul(currentWeights, currentInput);
        currentInput = matAdd(currentInput, currentBiases);
        currentInput = applyActivation(currentInput, sigmoid);
        this->activations.push_back(currentInput);
    }
}

void NeuralNet::backward(const Matrix &target, float learningRate) 
    {
        /***********************************************************************
        * 1) Retrieve the final output (post-activation) and the “previous”
        *    activation that fed into the final layer.
        ***********************************************************************/
        // The last element of 'activations' is our final layer's output
        cout << "activations size " << this->activations.size() <<endl;;
        Matrix output = this->activations.back();
        
        // The second to last element of 'activations' is the post-activation
        // of the layer before the final layer (i.e., input to the final layer).
        cout << "activations size " << this->activations.size() << endl;
        Matrix prevActivation = this->activations[this->activations.size() - 2];

        /***********************************************************************
        * 2) Compute the error at the final (output) layer using Mean Squared
        *    Error (MSE).
        *
        *    MSE formula (for one sample):
        *      L = 1/2 * Σ (output - target)²
        *
        *    dL/d(output) = (output - target) 
        *    (we omit the factor of 1/2 for simplicity in the gradient).
        *
        *    If using sigmoid at final layer, we multiply by derivative of
        *    sigmoid:  sigmoid'(z) = output * (1 - output).
        ***********************************************************************/
        // Create matrix deltaOutput to store the error signal at the output layer
        Matrix deltaOutput = output; // start by copying
        for (int i = 0; i < output.size(); i++) {
            for (int j = 0; j < output[0].size(); j++) {
                // error = (output - target)
                float error = output[i][j] - target[i][j];
                
                // derivative of sigmoid if final layer uses sigmoid
                float deriv = output[i][j] * (1.0f - output[i][j]);
                
                // final delta for output = error * sigmoid'(z)
                deltaOutput[i][j] = error * deriv;
            }
        }

        /***********************************************************************
        * 3) Compute gradients for the final layer's weights and biases:
        *
        *    - dW^(final) = deltaOutput * (prevActivation)^T
        *    - db^(final) = deltaOutput
        *
        *    Then perform the update step:
        *      W^(final) <- W^(final) - η * dW^(final)
        *      b^(final) <- b^(final) - η * db^(final)
        ***********************************************************************/
        // Grab the final layer weights & biases
        Matrix wLast = this->weights.back();
        Matrix bLast = this->biases.back();

        // We need prevActivation^T to do deltaOutput * prevActivation^T
        Matrix prevActivationT = transpose(prevActivation);

        // dW^(final) = deltaOutput x prevActivation^T
        Matrix dW_last = matMul(deltaOutput, prevActivationT);

        // Update final layer weights
        for (int r = 0; r < wLast.size(); r++) {
            for (int c = 0; c < wLast[0].size(); c++) {
                wLast[r][c] -= learningRate * dW_last[r][c];
            }
        }

        // Update final layer biases
        // Note: bLast and deltaOutput both have shape [outputLayerSize x 1]
        for (int r = 0; r < bLast.size(); r++) {
            bLast[r][0] -= learningRate * deltaOutput[r][0];
        }

        // Store back the updated weights/biases for the final layer
        this->weights.back() = wLast;
        this->biases.back()  = bLast;

        /***********************************************************************
        * 4) Propagate error backwards through the hidden layers:
        *
        *    For layer l, we have:
        *      delta^l = (W^(l+1)^T * delta^(l+1)) ⨀ sigmoid'(z^l)
        *
        *    Then, 
        *      dW^l = delta^l * (activation^(l-1))^T
        *      db^l = delta^l
        * 
        *    And we update W^l and b^l similarly.
        ***********************************************************************/
        // 'currentDelta' holds the delta for the layer we just handled (final layer).
        // We'll move it backward step by step.
        Matrix currentDelta = deltaOutput;

        // Traverse from the second-to-last layer index down to the first layer.
        // (i.e., from weights.size() - 2, down to 0)
        for (int layerIndex = (int)this->weights.size() - 2; layerIndex >= 0; layerIndex--)
        {
            // ---------------------------------------------------------
            // 4a) Compute the new delta for this hidden layer:
            //     newDelta = (W^(l+1).T * currentDelta) ⨀ derivOfSigmoid(activations[l+1])
            // ---------------------------------------------------------
            // W^(l+1):
            Matrix wNext = this->weights[layerIndex + 1];
            // transpose of that:
            Matrix wNextT = transpose(wNext);

            // Multiply: wNextT * currentDelta 
            Matrix newDelta = matMul(wNextT, currentDelta);

            // The post-activation for this layer is stored at activations[layerIndex+1].
            // Actually, that’s the “output” of layerIndex. 
            // We'll use it to get derivative of the sigmoid if we haven't stored z^l.
            Matrix currentActivation = this->activations[layerIndex + 1];

            // Now elementwise multiply by derivative of sigmoid: a * (1 - a).
            for (int i = 0; i < newDelta.size(); i++) {
                for (int j = 0; j < newDelta[0].size(); j++) {
                    float aVal = currentActivation[i][j];
                    float sigDeriv = aVal * (1.0f - aVal); 
                    newDelta[i][j] *= sigDeriv;
                }
            }

            // ---------------------------------------------------------
            // 4b) Compute gradients for W^l and b^l using newDelta
            // ---------------------------------------------------------
            // The "previous activation" for this layer is actually layerIndex’s activation 
            // (i.e., activations[layerIndex]).
            Matrix prevLayerActivation = this->activations[layerIndex];

            // We want dW^l = newDelta * prevLayerActivation^T
            Matrix prevLayerActivationT = transpose(prevLayerActivation);
            Matrix dW = matMul(newDelta, prevLayerActivationT);

            // Grab the current weights and biases for layerIndex
            Matrix wCurrent = this->weights[layerIndex];
            Matrix bCurrent = this->biases[layerIndex];

            // Update each weight
            for (int r = 0; r < wCurrent.size(); r++) {
                for (int c = 0; c < wCurrent[0].size(); c++) {
                    wCurrent[r][c] -= learningRate * dW[r][c];
                }
            }
            // Update each bias
            for (int r = 0; r < bCurrent.size(); r++) {
                bCurrent[r][0] -= learningRate * newDelta[r][0];
            }

            // Store back updated weights & biases
            this->weights[layerIndex] = wCurrent;
            this->biases[layerIndex]  = bCurrent;

            // ---------------------------------------------------------
            // 4c) Move 'currentDelta' backward to the next layer
            // ---------------------------------------------------------
            currentDelta = newDelta;
        }
    }   





    
    
