//
// Created by Mokhtar on 25/07/2025.
//
#include "utilities.h"
FilterSet conv_filters;
Vector1D conv_biases;
FilterSet conv2_filters;
Vector1D conv2_biases;
MatrixXf fc_weights;
Vector1D fc_biases;


int fc_input_size ;


float cross_entropy_loss(const Vector1D &y_predicted, const Vector1D &y_true) {
    if (y_predicted.size() != y_true.size()) {
        throw std::runtime_error("cross_entropy_loss: size mismatch");
    }

    const float epsilon = 1e-12f; // avoid log(0)
    float loss = 0.0f;

    for (size_t i = 0; i < y_predicted.size(); ++i) {
        float p = std::max(std::min(y_predicted[i], 1.0f - epsilon), epsilon);
        if(p == 0.0f) {
            cout << "error" << endl;
            cin.get();
        }
        loss -= y_true[i] * std::log(p);
    }

    return loss;
}

void initialize_matrix(MatrixXf &mat, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);

    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            mat(i, j) = static_cast<float>(dis(gen));
        }
    }
}

MatrixXf Initialize_fc_weights(int out_features, int in_features) {
    MatrixXf W(out_features, in_features);
    std::mt19937 gen(std::random_device{}());
    float bound = std::sqrt(3.0f / in_features);         // <— kaiming_uniform_
    std::uniform_real_distribution<float> dis(-bound, bound);
    for (int i = 0; i < out_features; ++i)
        for (int j = 0; j < in_features; ++j)
            W(i, j) = dis(gen);
    return W;
}



// Initialize filters
// He-uniform (Kaiming uniform) initialization for Conv filters
void initialize_filters(FilterSet& filters, int F, int C, int KH, int KW) {
    std::mt19937 gen(std::random_device{}());
    int fan_in = C * KH * KW;
    float bound = std::sqrt(3.0f / fan_in);              // <— match kaiming_uniform_

    std::uniform_real_distribution<float> dis(-bound, bound);
    filters.resize(F);
    for (int f = 0; f < F; ++f) {
        filters[f].resize(C);
        for (int c = 0; c < C; ++c) {
            filters[f][c].resize(KH, KW);
            for (int i = 0; i < KH; ++i)
                for (int j = 0; j < KW; ++j)
                    filters[f][c](i, j) = dis(gen);
        }
    }
}
// Bias initialization (PyTorch: U(-1/sqrt(fan_in), 1/sqrt(fan_in)))
void initialize_biases(Vector1D &biases,
                       int size,          // out_channels
                       int in_channels,   // in_channels
                       int filter_height,
                       int filter_width) {
    std::random_device rd;
    std::mt19937 gen(rd());

    int fan_in = in_channels * filter_height * filter_width;
    float bound = 1.0f / std::sqrt(fan_in);

    std::uniform_real_distribution<float> dis(-bound, bound);

    biases.resize(size);
    for (auto &b : biases) {
        //b = dis(gen);
        b = 0 ;
    }
}

// Example usage inside your function before training loop:
Tensor conv2D(const Tensor& input,
              const FilterSet& filters,
              const Vector1D& biases,
              int stride,
              const int & padding) {
    int C = input.size();                     // Input channels
    int H = input[0].rows(), W = input[0].cols();
    int F = filters.size();                   // Number of filters
    int KH = filters[0][0].rows(), KW = filters[0][0].cols();

    // ---- Step 0: Apply zero padding ----
    Tensor padded_input(C, Matrix2D(H + 2 * padding, W + 2 * padding));
    for (int c = 0; c < C; ++c) {
        padded_input[c].setZero();
        padded_input[c].block(padding, padding, H, W) = input[c];
    }

    int padded_H = padded_input[0].rows();
    int padded_W = padded_input[0].cols();

    // ---- Step 1: Output size ----
    int outH = (padded_H - KH) / stride + 1;
    int outW = (padded_W - KW) / stride + 1;

    // ---- Step 2: im2col ----
    int patch_size = C * KH * KW;
    int num_patches = outH * outW;
    Eigen::MatrixXf cols(patch_size, num_patches);

    int col_idx = 0;
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            int row_offset = i * stride;
            int col_offset = j * stride;

            int idx = 0;
            for (int c = 0; c < C; ++c) {
                auto patch = padded_input[c].block(row_offset, col_offset, KH, KW);
                Eigen::Map<const Eigen::VectorXf> patch_vec(patch.data(), KH * KW);
                cols.block(idx, col_idx, KH * KW, 1) = patch_vec;
                idx += KH * KW;
            }
            col_idx++;
        }
    }

    // ---- Step 3: reshape filters ----
    Eigen::MatrixXf filter_matrix(F, patch_size);
    for (int f = 0; f < F; ++f) {
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            Eigen::Map<const Eigen::VectorXf> filt_vec(filters[f][c].data(), KH * KW);
            filter_matrix.block(f, idx, 1, KH * KW) = filt_vec.transpose();
            idx += KH * KW;
        }
    }

    // ---- Step 4: Matrix multiply ----
    Eigen::MatrixXf result = filter_matrix * cols;

    // ---- Step 5: Add biases ----
    for (int f = 0; f < F; ++f) {
        result.row(f).array() += biases(f);
    }

    // ---- Step 6: Reshape into Tensor (fixed layout) ----
    Tensor output(F, Matrix2D(outH, outW));
    for (int f = 0; f < F; ++f) {
        int col_idx = 0;
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                output[f](i, j) = result(f, col_idx++);
            }
        }
    }

    return output;
}





void relu(Tensor& tensor) {
    for (auto& mat : tensor)
        mat = mat.unaryExpr([](float x) { return std::max(0.0f, x); });
}



MaxPoolResult maxPool2x2(const Tensor& input) {
    int C = input.size();
    int H = input[0].rows(), W = input[0].cols();
    int outH = H / 2, outW = W / 2;

    Tensor pooled(C, Matrix2D(outH, outW));
    Tensor argmax_i(C, Matrix2D::Zero(outH, outW));
    Tensor argmax_j(C, Matrix2D::Zero(outH, outW));

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                int base_i = i * 2, base_j = j * 2;
                float max_val = -std::numeric_limits<float>::infinity();
                int best_i = base_i, best_j = base_j;

                // Search 2×2 window
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        int r = base_i + di, col = base_j + dj;
                        float v = input[c](r, col);
                        if (v > max_val) {
                            max_val = v;
                            best_i = r;
                            best_j = col;
                        }
                    }
                }
                pooled[c](i, j)   = max_val;
                argmax_i[c](i, j) = best_i;
                argmax_j[c](i, j) = best_j;
            }
        }
    }
    MaxPoolResult result;
    result.pooled = pooled;
    result.argmax_i = argmax_i;
    result.argmax_j = argmax_j;
    return result;
}


Vector1D flatten(const Tensor& input, const int fc_input_size) {
    Vector1D flat(fc_input_size);
    int pos = 0;

    for (const auto& mat : input) {
        // Eigen::Map respects the storage order (column-major by default)
        Eigen::Map<const Vector1D> vec(mat.data(), mat.size());
        flat.segment(pos, mat.size()) = vec;
        pos += mat.size();
    }

    return flat;
}


Vector1D fullyConnected(const Vector1D& input, const MatrixXf& weights, const Vector1D& bias) {
    return (weights * input + bias);
}

Vector1D softmax(const Vector1D& input) {
    float maxVal = input.maxCoeff();
    Vector1D exps = (input.array() - maxVal).exp();
    return exps / exps.sum();
}





void relu_backward(Tensor& dOut, const Tensor& preact) {
    for (int c = 0; c < (int)dOut.size(); ++c) {
        // Mask: 1 where preact > 0, 0 otherwise
        dOut[c] = dOut[c].array() * (preact[c].array() > 0).cast<float>();
    }
}


Tensor maxPool2x2_backward(const Tensor& dPooled,
                           const MaxPoolResult& cache,
                           int inputH, int inputW) {
    int C = dPooled.size();
    Tensor dInput(C, Matrix2D::Zero(inputH, inputW));

    for (int c = 0; c < C; ++c) {
        int outH = dPooled[c].rows(), outW = dPooled[c].cols();
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                int r   = static_cast<int>(cache.argmax_i[c](i, j));
                int col = static_cast<int>(cache.argmax_j[c](i, j));
                dInput[c](r, col) = dPooled[c](i, j);
            }
        }
    }
    return dInput;
}

// Vectorized conv2D backward (computes dFilters, dBiases, dInput)
// input:     Tensor of size C x H x W
// dOut:      Tensor of size F x outH x outW  (gradients wrt conv outputs)
// filters:   FilterSet of size F x C x KH x KW
// dFilters:  output FilterSet (same shape as filters) - gradients wrt filters
// dBiases:   output Vector1D of length F - gradients wrt biases
// dInput:    output Tensor of size C x H x W - gradient wrt input
// stride:    convolution stride (assumed same as forward)
// Shapes:
// input:   C x H x W
// filters: F x C x KH x KW
// dOut:    F x outH x outW   (same outH/outW as forward)
// stride, padding: same as forward
void conv2D_backward(const Tensor& input,
                     const Tensor& dOut,
                     const FilterSet& filters,
                     FilterSet& dFilters,
                     Vector1D& dBiases,
                     Tensor& dInput,
                     int stride,
                     int padding)
{
    using Eigen::MatrixXf;
    using Eigen::VectorXf;

    const int C  = static_cast<int>(input.size());
    const int H  = input[0].rows();
    const int W  = input[0].cols();

    const int F  = static_cast<int>(filters.size());
    const int KH = filters[0][0].rows();
    const int KW = filters[0][0].cols();

    const int outH = dOut[0].rows();
    const int outW = dOut[0].cols();

    const int patch_size  = C * KH * KW;
    const int num_patches = outH * outW;

    // --- 1) Pad input (same as forward) ---
    const int padded_H = H + 2 * padding;
    const int padded_W = W + 2 * padding;
    Tensor padded_input(C, Matrix2D::Zero(padded_H, padded_W));
    for (int c = 0; c < C; ++c) {
        padded_input[c].block(padding, padding, H, W) = input[c];
    }

    // --- 2) Build cols (patch_size x num_patches) exactly like forward ---

    MatrixXf cols(patch_size, num_patches);
    int col_idx = 0;
    for (int i = 0; i < outH; ++i) {
        const int row_offset = i * stride;
        for (int j = 0; j < outW; ++j) {
            const int col_offset = j * stride;

            int idx = 0;
            for (int c = 0; c < C; ++c) {
                auto patch = padded_input[c].block(row_offset, col_offset, KH, KW);
                Eigen::Map<const Eigen::VectorXf> patch_vec(patch.data(), KH * KW); // column-major
                cols.block(idx, col_idx, KH * KW, 1) = patch_vec;
                idx += KH * KW;
            }
            ++col_idx;
        }
    }


    // --- 3) Flatten dOut into (F x num_patches) with SAME col_idx order ---
    MatrixXf dOut_mat(F, num_patches);
    col_idx = 0;
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            for (int f = 0; f < F; ++f) {
                dOut_mat(f, col_idx) = dOut[f](i, j);
            }
            ++col_idx;
        }
    }

    // --- 4) dBiases: sum over spatial dims for each filter ---
    dBiases = Vector1D::Zero(F);
    dBiases = dOut_mat.rowwise().sum();


    // --- 5) Build filter_matrix (F x patch_size) in the SAME order as forward ---
    MatrixXf filter_matrix(F, patch_size);
    for (int f = 0; f < F; ++f) {
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            Eigen::Map<const Eigen::VectorXf> fv(filters[f][c].data(), KH * KW); // column-major
            filter_matrix.block(f, idx, 1, KH * KW) = fv.transpose();
            idx += KH * KW;
        }
    }


    // --- 6) dFilters_mat = dOut_mat * cols^T  (F x patch_size) ---
    MatrixXf dFilters_mat = dOut_mat * cols.transpose();

    // --- 7) Unpack dFilters_mat back into FilterSet ---
    dFilters.clear();
    dFilters.resize(F);
    for (int f = 0; f < F; ++f) {
        dFilters[f].resize(C);
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            dFilters[f][c] = Matrix2D::Zero(KH, KW);
            for (int kw = 0; kw < KW; ++kw)
                for (int kh = 0; kh < KH; ++kh)   // note kh inner for column-major stride
                    dFilters[f][c](kh, kw) = dFilters_mat(f, idx++);
        }
    }


    // --- 8) dCols = filter_matrix^T * dOut_mat  (patch_size x num_patches) ---
    MatrixXf dCols = filter_matrix.transpose() * dOut_mat;

    // --- 9) col2im: accumulate into padded_dInput (C x padded_H x padded_W) ---
    Tensor padded_dInput(C, Matrix2D::Zero(padded_H, padded_W));

    col_idx = 0;
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            const int row_offset = i * stride;
            const int col_offset = j * stride;

            int idx = 0; // walks over the patch_size for this column
            for (int c = 0; c < C; ++c) {
                for (int ph = 0; ph < KH; ++ph) {
                    for (int pw = 0; pw < KW; ++pw) {
                        padded_dInput[c](row_offset + ph, col_offset + pw) += dCols(idx++, col_idx);
                    }
                }
            }
            ++col_idx;
        }
    }

    // --- 10) Crop padding to get dInput (C x H x W) ---
    dInput.clear();
    dInput.resize(C);
    for (int c = 0; c < C; ++c) {
        dInput[c] = padded_dInput[c].block(padding, padding, H, W);
    }
}


Tensor unflatten(const Vector1D& flat, const Tensor& reference) {
    Tensor output;
    output.reserve(reference.size());

    int offset = 0;
    for (const auto& channel : reference) {
        int rows = channel.rows();
        int cols = channel.cols();

        // Map back the flat vector to column-major matrix
        Eigen::Map<const Eigen::MatrixXf> mat(&flat[offset], rows, cols);
        output.push_back(Eigen::MatrixXf(mat)); // copy into own memory
        offset += rows * cols;
    }

    return output;
}



Vector1D cnn_forward(const Tensor& image,
                     MaxPoolResult & pooled1,
                     Tensor & conv1_out,
                     MaxPoolResult & pooled2,
                     Tensor & conv2_out,
                     Vector1D & flat, const int & filter_height, const int fc_input_size) {
    // Conv layer 1
    int padding = (filter_height - 1) / 2;  // for stride=1
    conv1_out = conv2D(image, conv_filters, conv_biases, 1, padding);
    relu(conv1_out);
    pooled1 = maxPool2x2(conv1_out);
    // Conv layer 2
    conv2_out = conv2D(pooled1.pooled, conv2_filters, conv2_biases, 1, padding);

    relu(conv2_out);
    pooled2 = maxPool2x2(conv2_out);
    // Flatten after second pool
    flat = flatten(pooled2.pooled,fc_input_size);
    // FC + softmax
    Vector1D fc_out = fullyConnected(flat, fc_weights, fc_biases);
    // insert in cnn_forward right before return softmax(fc_out)

    return softmax(fc_out);
}



Gradients cnn_backward(const Tensor& image,
                       const Tensor& conv1_out,
                       const MaxPoolResult & pooled1,
                       const Tensor& conv2_out,
                       const MaxPoolResult & pooled2,
                       const Vector1D& flat,
                       const Vector1D& fc_out,
                       const Vector1D& y_true,
                       const FilterSet& conv1_filters,
                       const FilterSet& conv2_filters,
                       const MatrixXf& fc_weights, const int & padding)
{
    Gradients grads;

    // 1. Softmax + CrossEntropy
    Vector1D probs = fc_out;
    Vector1D dZ_fc(probs.size());
    for (int i = 0; i < dZ_fc.size(); ++i)
        dZ_fc[i] = probs[i] - y_true[i];

    // 2. Fully connected backward
    grads.dFcWeights = dZ_fc * flat.transpose();
    grads.dFcBiases = dZ_fc;
    Vector1D dFlat = fc_weights.transpose() * dZ_fc;
    // 3. Unflatten to pooled2 shape
    Tensor dPooled2 = unflatten(dFlat, pooled2.pooled);
    // 4. MaxPool2 backward (2nd pooling)
    Tensor dConv2_out = maxPool2x2_backward(dPooled2, pooled2, conv2_out[0].rows(), conv2_out[0].cols());
    // 5. ReLU backward (2nd conv)
    relu_backward(dConv2_out, conv2_out);

    // 6. Conv2 backward
    FilterSet dConv2Filters;
    Vector1D dConv2Biases;
    Tensor dPooled1;
    conv2D_backward(pooled1.pooled, dConv2_out, conv2_filters,
                    dConv2Filters, dConv2Biases, dPooled1, 1, padding);

    // 7. MaxPool2 backward (1st pooling)
    Tensor dConv1_out = maxPool2x2_backward(dPooled1, pooled1, conv1_out[0].rows(), conv1_out[0].cols());

    // 8. ReLU backward (1st conv)
    relu_backward(dConv1_out, conv1_out);

    // 9. Conv1 backward
    FilterSet dConv1Filters;
    Vector1D dConv1Biases;
    Tensor dInput;
    conv2D_backward(image, dConv1_out, conv1_filters,
                    dConv1Filters, dConv1Biases, dInput, 1, padding);


    // Store gradients
    grads.dConvFilters = dConv1Filters; // for conv1

    grads.dConvBiases = dConv1Biases;

    grads.dConv2Filters = dConv2Filters; // you'll add to Gradients struct

    grads.dConv2Biases = dConv2Biases;
    //cin.get();
    return grads;
}


void PerformConvNN(const vector<Tensor> & input, const vector<int> & labels, const vector<string> & path_names, int batch_size) {
    float lr = 0.001f;
    int num_filters1 = 16;
    int num_filters2 = 32;
    int channels = 3;
    int filter_height = 3;
    int filter_width = 3;
    int image_height = 32;
    int image_width = 32;
    int padding = 1; // SAME padding

    // --- Gradient sanity check ---

    // First conv layer
    initialize_filters(conv_filters, num_filters1, channels, filter_height, filter_width);
    initialize_biases(conv_biases, num_filters1, channels, filter_height, filter_width);
    // After first conv + pool
    int conv1_h = (image_height + 2 * padding - filter_height) / 1 + 1;
    int conv1_w = (image_width  + 2 * padding - filter_width)  / 1 + 1;
    int pooled1_h = conv1_h / 2; // maxpool 2x2, stride=2
    int pooled1_w = conv1_w / 2;

    // Second conv layer
    initialize_filters(conv2_filters, num_filters2, num_filters1, filter_height, filter_width);
    initialize_biases(conv2_biases, num_filters2, num_filters1, filter_height, filter_width);

    // After second conv + pool
    int conv2_h = (pooled1_h + 2 * padding - filter_height) / 1 + 1;
    int conv2_w = (pooled1_w + 2 * padding - filter_width)  / 1 + 1;
    int pooled2_h = conv2_h / 2;
    int pooled2_w = conv2_w / 2;


    // Fully connected layer
    fc_input_size = pooled2_h * pooled2_w * num_filters2;
    int fc_output_size = 10; // CIFAR-10
    fc_weights.resize(fc_output_size, fc_input_size);
    fc_biases.resize(fc_output_size);
    fc_weights = Initialize_fc_weights(fc_output_size, fc_input_size);
    initialize_biases(fc_biases, fc_output_size, fc_input_size, 1, 1);

    // momentum variables
    float momentum = 0.9f;

    // Initialize velocity variables with same shapes as weights and grads
    MatrixXf v_fc_weights = MatrixXf::Zero(fc_weights.rows(), fc_weights.cols());
    Vector1D v_fc_biases = Vector1D::Zero(fc_biases.size());

    FilterSet v_conv_filters(conv_filters.size(),
        vector<MatrixXf>(conv_filters[0].size(),
        MatrixXf::Zero(conv_filters[0][0].rows(), conv_filters[0][0].cols())));

    Vector1D v_conv_biases = Vector1D::Zero(conv_biases.size());

    FilterSet v_conv2_filters(conv2_filters.size(),
        vector<MatrixXf>(conv2_filters[0].size(),
        MatrixXf::Zero(conv2_filters[0][0].rows(), conv2_filters[0][0].cols())));

    Vector1D v_conv2_biases = Vector1D::Zero(conv2_biases.size());
    //--------------
    int num_samples = input.size();
    vector<int> indices(input.size());
    iota(indices.begin(), indices.end(), 0);
    //shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
    // ---- Training ----
    for (int epoch = 0; epoch < 10; ++epoch) {
        // shuffle the data
        shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        float total_loss = 0.0f;
        int correct = 0;

        for (int start = 0; start < num_samples; start += batch_size) {

            int end = min(start + batch_size, num_samples);
            int current_batch_size = end - start;
            // Accumulate Gradients
            Gradients batch_grads;
            batch_grads.dFcWeights = MatrixXf::Zero(fc_weights.rows(), fc_weights.cols());
            batch_grads.dFcBiases = Vector1D::Zero(fc_biases.size());
            batch_grads.dConvFilters = FilterSet(
                conv_filters.size(),
                vector<MatrixXf>(conv_filters[0].size(),
                                 MatrixXf::Zero(conv_filters[0][0].rows(), conv_filters[0][0].cols())));
            batch_grads.dConvBiases = Vector1D::Zero(conv_biases.size());
            batch_grads.dConv2Filters = FilterSet(
                conv2_filters.size(),
                vector<MatrixXf>(conv2_filters[0].size(),
                                 MatrixXf::Zero(conv2_filters[0][0].rows(), conv2_filters[0][0].cols())));
            batch_grads.dConv2Biases = Vector1D::Zero(conv2_biases.size());
            //________________________

            for (int im = start; im < end; ++im) {

                Tensor image = input[indices[im]];
                MaxPoolResult pooled,pooled2;
                Tensor  conv_out, conv2_out;
                Vector1D flat;

                // Forward Pass
                Vector1D y_predicted = cnn_forward(image, pooled, conv_out,
                     pooled2, conv2_out,flat, filter_height,fc_input_size) ;

                for (float val : y_predicted) {
                    if (std::isnan(val) || std::isinf(val)) {
                        cout << "Invalid value in y_predicted\n";
                        cin.get();
                    }
                }
                // Ground truth
                Vector1D y_true = Vector1D::Zero(y_predicted.size());
                y_true[labels[indices[im]]] = 1.0f;
                Gradients grads = cnn_backward(image,conv_out,pooled,conv2_out,pooled2,
                    flat,y_predicted,y_true,conv_filters,conv2_filters,fc_weights,padding);

                // Accumulate Gradients

                batch_grads.dFcWeights += grads.dFcWeights;
                batch_grads.dFcBiases += grads.dFcBiases;

                // first convolution
                for (size_t f = 0; f < conv_filters.size(); ++f) {
                    for (size_t c = 0; c < conv_filters[f].size(); ++c) {
                        batch_grads.dConvFilters[f][c] += grads.dConvFilters[f][c];
                    }
                    batch_grads.dConvBiases[f] += grads.dConvBiases[f];
                }
                // second convolution
                for (size_t f = 0; f < conv2_filters.size(); ++f) {
                    for (size_t c = 0; c < conv2_filters[f].size(); ++c) {
                        batch_grads.dConv2Filters[f][c] += grads.dConv2Filters[f][c];
                    }
                    batch_grads.dConv2Biases[f] += grads.dConv2Biases[f];
                }

                // Track Loss and Accuracy
                total_loss += cross_entropy_loss(y_predicted, y_true);
                if(isnan(total_loss)) {
                    cout << "error ";
                    cin.get();
                }
                int pred_label;
                y_predicted.maxCoeff(&pred_label);

                if (pred_label == labels[indices[im]]) correct++;

            }

            // ---- Momentum SGD update ----

            v_fc_weights = momentum * v_fc_weights - (lr / current_batch_size) * batch_grads.dFcWeights;
            fc_weights += v_fc_weights;

            v_fc_biases = momentum * v_fc_biases - (lr / current_batch_size) * batch_grads.dFcBiases;
            fc_biases += v_fc_biases;

            for (size_t f = 0; f < conv_filters.size(); ++f) {
                for (size_t c = 0; c < conv_filters[f].size(); ++c) {
                    v_conv_filters[f][c] = momentum * v_conv_filters[f][c] - (lr / current_batch_size) * batch_grads.dConvFilters[f][c];
                    conv_filters[f][c] += v_conv_filters[f][c];
                }
                v_conv_biases[f] = momentum * v_conv_biases[f] - (lr / current_batch_size) * batch_grads.dConvBiases[f];
                conv_biases[f] += v_conv_biases[f];
            }

            for (size_t f = 0; f < conv2_filters.size(); ++f) {
                for (size_t c = 0; c < conv2_filters[f].size(); ++c) {
                    v_conv2_filters[f][c] = momentum * v_conv2_filters[f][c] - (lr / current_batch_size) * batch_grads.dConv2Filters[f][c];
                    conv2_filters[f][c] += v_conv2_filters[f][c];
                }
                v_conv2_biases[f] = momentum * v_conv2_biases[f] - (lr / current_batch_size) * batch_grads.dConv2Biases[f];
                conv2_biases[f] += v_conv2_biases[f];
            }

        }

        // Epoch summary
        //lr = lr*0.9;
        float avg_loss = total_loss / num_samples;
        float accuracy = (float)correct / num_samples;
        cout << "Epoch " << epoch
             << " | Loss: " << avg_loss
             << " | Accuracy: " << accuracy * 100.0f << "%" << endl;
    }


}

// Evaluate on the test images

void Evaluate_model(const vector<Tensor> &test_images, vector<int> &test_labels) {
    int correct = 0;
    MaxPoolResult pooled, pooled2;
    Tensor  conv_out;
    Tensor  conv2_out;
    Vector1D flat;
    int filter_height = 3 ;
    // Forward Pass
    for(int im = 0; im < test_images.size(); ++im) {
        Vector1D y_predicted = cnn_forward(test_images[im], pooled, conv_out,
                     pooled2, conv2_out,flat, filter_height,fc_input_size) ;
        int y_pred ;
        y_predicted.maxCoeff(&y_pred);
        if(test_labels[im] == y_pred) {
            correct++;
        }
    }

    cout << "Accuracy: " << correct * 100.0f / test_images.size() << " correct " << correct << endl;

}


