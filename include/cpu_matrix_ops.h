#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

// we will define a matrix as a vector of vectors of floats
typedef std::vector<std::vector<float> > Matrix;

Matrix matMul(const Matrix &a, const Matrix &b);

Matrix matAdd(const Matrix &a, const Matrix &b);

Matrix transpose(const Matrix &a);

Matrix applyActivation(const Matrix &a, float (*activation)(float));

float sigmoid(float x);

#endif