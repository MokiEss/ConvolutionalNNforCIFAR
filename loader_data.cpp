//
// Created by Mokhtar on 27/08/2025.
//

#include "loader_data.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Tensor load_image_as_tensor(const string& image_path) {
    int width, height, channels;
    unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 3); // Force 3 channels (RGB)

    if (!img) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return {};
    }

    Tensor image_tensor(3, MatrixXf(height, width)); // Channels-first (C x H x W)

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = (i * width + j) * 3 + c; // Interleaved RGB
                image_tensor[c](i, j) = static_cast<float>(img[idx]) / 127.5f - 1.0f; // normalize [-1 1]
                if(image_tensor[c](i, j) <-1 || image_tensor[c](i, j) > 1) {
                    cout << "error " ;
                    cin.get();
                }

            }

        }

    }

    stbi_image_free(img);
    return image_tensor;
}




int determine_label(const string & name) {
    if(name == "airplane") return 0;
    if(name == "automobile") return 1;
    if(name == "bird") return 2;
    if(name == "cat") return 3;
    if(name == "deer") return 4;
    if(name == "dog") return 5;
    if(name == "frog") return 6;
    if(name == "horse") return 7;
    if(name == "ship") return 8;
    if(name == "truck") return 9;
    else return -1 ;
}


vector<Tensor> load_image_as_tensors(string& base_dir, vector<int>& labels,vector<string>& image_path) {
    vector<Tensor> image_tensors;
    int i = 0 ;
    for (const auto& class_dir : fs::directory_iterator(base_dir)) {
        if (!fs::is_directory(class_dir)) continue;
        int label = determine_label(class_dir.path().filename()) ;
        for (const auto& image_file : fs::directory_iterator(class_dir)) {
            auto pixels = load_image_as_tensor(image_file.path().string());
            image_tensors.push_back(pixels);
            labels.push_back(label);
            image_path.push_back(image_file.path().string());
        }
    }
    cout << "Loaded images " << image_tensors.size()  << endl;
    return image_tensors;
}