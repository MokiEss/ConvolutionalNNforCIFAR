# Convolutional Neural Network in C++ (Learning Project)

This project implements a **Convolutional Neural Network (CNN)** from scratch in **C++**, using the [Eigen](https://eigen.tuxfamily.org/) linear algebra library.  

The CNN consists of:
- **Two convolutional layers**
- **One fully connected layer**

⚠️ **Note:** This project is for **learning purposes only**. It is not optimized for performance or production use, but rather serves as a way to understand the inner workings of convolutional neural networks and how they can be implemented without deep learning frameworks.

---

## Features
- Forward pass through convolutional layers with basic filters
- Activation functions
- Fully connected output layer
- Example training data loader for CIFAR-10 images (for experimentation)
- Uses Eigen for efficient matrix operations

---

## CNN Architecture (Simplified)
Input Image (e.g., CIFAR-10 32x32 RGB) -> Convolution Layer 1 -> Convolution Layer 2 -> Flatten -> Fully Connected Layer -> Classification Output

---

## Requirements
- **C++20** (or newer)
- **CMake 3.29+**
- **Eigen 3.4.0** (included in the project folder or installed separately)


---

## Building the Project
Clone the repository and build using CMake:

```bash
git clone https://github.com/MokiEss/ConvolutionalNNforCIFAR.git
cd ConvolutionalNNforCIFAR
mkdir build && cd build
cmake ..
make
```
---
## Running
After building, you can run the program with:
```bash
./ConvolutionalNNforCIFAR
```
---
## Notes
This is a learning project. The CNN implementation is simplified and not optimized for speed or large-scale datasets.
The CIFAR-10 dataset is not included. You can download it separately and adjust paths in the code if needed.
The project is meant to provide insight into how neural networks can be implemented from scratch in C++, rather than to compete with established frameworks like TensorFlow or PyTorch.
