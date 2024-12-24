#include "../include/cpu_matrix_ops.h"
#include <vector>
#include <iostream>

Matrix matMul(const Matrix &a, const Matrix &b) {
    int aRows = a.size(), aCols = a[0].size();
    int bRows = b.size(), bCols = b[0].size();

    if (aCols != bRows) {
        throw "Matrix dimensions are incompatible for multiplication";
    }

    Matrix result(aRows, std::vector<float>(bCols, 0));

    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            for (int k = 0; k < aCols; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

Matrix matAdd(const Matrix &a, const Matrix &b) {
    int aRows = a.size(), aCols = a[0].size();
    int bRows = b.size(), bCols = b[0].size();

    if (aRows != bRows || aCols != bCols) {
        throw "Matrix dimensions are incompatible for multiplication";
    }

    Matrix result(aRows, std::vector<float>(bCols, 0));

    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < aCols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        } 
    }

    return result;
}

Matrix transpose(const Matrix &a) {
    int rows = a.size(), cols = a[0].size();

    Matrix result(cols, std::vector<float>(rows, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = a[i][j];
        }
    }

    return result;
}

Matrix applyActivation(const Matrix &a, float (*activation)(float)) {
    int rows = a.size(), cols = a[0].size();

    Matrix result(rows, std::vector<float>(cols, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = activation(a[i][j]);
        }
    }

    return result;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}
