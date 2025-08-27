#ifndef UTILITIES_H
#define UTILITIES_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>

#endif //UTILITIES_H
using namespace std;
using namespace Eigen ;


using Matrix2D = Eigen::MatrixXf;
using Vector1D = Eigen::VectorXf;
using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TensorRow = std::vector<MatrixRM>;
// Tensor = vector of matrices (channels)
using Tensor = vector<Matrix2D>;            // 3D Tensor: C x H x W
using FilterSet = vector<Tensor>;         // N filters (each with C channels)
using Input_data = vector<Tensor>;
struct MaxPoolResult {
    Tensor pooled;      // pooled output (C × H/2 × W/2)
    Tensor argmax_i;    // row indices of chosen max
    Tensor argmax_j;    // col indices of chosen max
};

Tensor conv2D(const Tensor& input,
              const FilterSet& filters,
              const Vector1D& biases,
              int stride,
              const int & padding);

Vector1D softmax(const Vector1D& input);
Vector1D fullyConnected(const Vector1D& input, const MatrixXf& weights, const Vector1D& bias);

MaxPoolResult maxPool2x2(const Tensor& input);
void relu(Tensor& tensor);


struct Gradients {
    FilterSet dConvFilters;
    Vector1D dConvBiases;
    MatrixXf dFcWeights;
    Vector1D dFcBiases;
    FilterSet dConv2Filters; // you'll add to Gradients struct
    Vector1D dConv2Biases;
    Tensor dInput;
};

void PerformConvNN(const vector<Tensor> & train_images, const vector<int> & train_labels,const vector<string> & path_names, int batch_size = 64) ;
void Evaluate_model(const vector<Tensor> &test_images, vector<int> &test_labels) ;