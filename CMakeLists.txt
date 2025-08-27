cmake_minimum_required(VERSION 3.29)
project(ConvolutionalNNforCIFAR)

set(CMAKE_CXX_STANDARD 20)

# Add Eigen include directory
include_directories(eigen-3.4.0/)
include_directories(/opt/homebrew/include)

# Link library search path
link_directories(/opt/homebrew/lib)

# Source files
set(SOURCES
        main.cpp
        convolution.cpp
        loader_data.cpp
)

# Create the executable
add_executable(ConvolutionalNNforCIFAR ${SOURCES})

# Link libraries
target_link_libraries(ConvolutionalNNforCIFAR png)
