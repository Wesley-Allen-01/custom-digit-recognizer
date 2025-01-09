#include <iostream>
#include <vector>
#include <cmath>      // for fabs
#include <cassert>    // for assert if you want
#include "../include/neuralNet.h"
#include "../include/cpu_matrix_ops.h"


// Bring everything from std into the global namespace for brevity.
using namespace std;

int main() {
    // 1. Construct a small neural network:
    //    - Input size = 2
    //    - One hidden layer of size 2
    //    - Output size = 1
    NeuralNet net(2, {2}, 1);

    // 2. Define a single training sample (input + target):
    //    Let's pretend we want the net to output 1.0 for input [0.5, 0.3].
    Matrix input = {
        {0.5f},{0.3f}  
    };
    Matrix target = {
        {1.0f}   // we want the network to produce ~1.0
    };

    // 3. Run the forward pass to get the initial output
    net.forward(input);
    Matrix initialOutput = net.getActivations().back();
    float initialVal = initialOutput[0][0];  // It's a 1x1 matrix

    // Print out the initial output
    cout << "Initial output: " << initialVal << endl;
    
    // 4. Perform one backward pass (gradient descent step)
    float learningRate = 0.1f;   // choose something noticeable
    net.backward(target, learningRate);
    cout << "Passed the first backward pass" << endl;
    // 5. Forward pass again to see how output has changed
    net.forward(input);
    Matrix newOutput = net.getActivations().back();
    float newVal = newOutput[0][0];
    cout << "Passed the first forward pass" << endl;
    // Print out the output after backprop
    cout << "New output after one backprop step: " << newVal << endl;

    // 6. Check difference
    //    Usually, you'd expect the new output to be a bit closer to target
    //    than the initial output—assuming everything is set up correctly.
    float initialError = fabs(initialVal - 1.0f);
    float newError     = fabs(newVal - 1.0f);

    cout << "Distance to target before: " << initialError 
         << "\nDistance to target after:  " << newError << endl;

    // Optional: If you want to automatically pass/fail the test,
    // you can assert that newError < initialError:
    // (This is naive but helpful for a quick sanity check.)
    if (newError < initialError) {
        cout << "[PASS] The output moved closer to the target.\n";
    } else {
        cout << "[FAIL] The output did not move closer to the target.\n";
    }

    return 0;
}
