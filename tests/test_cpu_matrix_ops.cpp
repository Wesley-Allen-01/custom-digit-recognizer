#include "../include/cpu_matrix_ops.h"
#include <iostream>
#include <stdexcept>
#include <vector>

void testMatrixMultiply() {
    // Define test matrices
    Matrix A = {{1, 2, 3}, {4, 5, 6}};
    Matrix B = {{7, 8}, {9, 10}, {11, 12}};

    // Expected result of A * B
    Matrix expected = {{58, 64}, {139, 154}};

    // Perform matrix multiplication
    try {
        Matrix result = matMul(A, B);

        // Verify the result
        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                if (result[i][j] != expected[i][j]) {
                    std::cerr << "Test Failed: Element at (" << i << ", " << j << ") is incorrect.\n";
                    return;
                }
            }
        }
        std::cout << "Test Passed: Matrix Multiply\n";
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: Exception thrown - " << e.what() << "\n";
    }
}

void testMatrixAddition() {
    // Define test matrices
    Matrix A = {{1, 2, 3}, {4, 5, 6}};
    Matrix B = {{7, 8, 9}, {10, 11, 12}};

    // Expected result of A + B
    Matrix expected = {{8, 10, 12}, {14, 16, 18}};

    // Perform matrix addition
    try {
        Matrix result = matAdd(A, B);

        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                if (result[i][j] != expected[i][j]) {
                    std::cerr << "Test Failed: Element at (" << i << ", " << j << ") is incorrect.\n";
                    return;
                }
            }
        }
        std::cout << "Test Passed: Matrix Addition\n";
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: Exception thrown - " << e.what() << "\n";
    }
}

void testMatrixTranspose() {
    // Define test matrix
    Matrix A = {{1, 2, 3}, {4, 5, 6}};

    // Expected result of transpose(A)
    Matrix expected = {{1, 4}, {2, 5}, {3, 6}};

    // Perform matrix transpose
    try {
        Matrix result = transpose(A);

        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                if (result[i][j] != expected[i][j]) {
                    std::cerr << "Test Failed: Element at (" << i << ", " << j << ") is incorrect.\n";
                    return;
                }
            }
        }
        std::cout << "Test Passed: Matrix Transpose\n";
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: Exception thrown - " << e.what() << "\n";
    }
}

void testApplyActivation() {
    Matrix A = {{0}, {1}, {2}};
    Matrix expected = {{0.5}, {0.731059}, {0.880797}};

    try {
        Matrix result = applyActivation(A, sigmoid);

        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                if ((result[i][j] - expected[i][j]) > 0.0001) {
                    std::cerr << "Test Failed: Element at (" << i << ", " << j << ") is incorrect.\n";
                    return;
                }
            }
        }
        std::cout << "Test Passed: Apply Activation\n";
    } catch (const std::exception& e) {
        std::cerr << "Test Failed: Exception thrown - " << e.what() << "\n";
    }
}

int main() {
    std::cout << "Running tests" << std::endl;
    testMatrixMultiply();
    testMatrixAddition();
    testMatrixTranspose();
    testApplyActivation();
    std::cout << "Tests complete" << std::endl;
    return 0;
}