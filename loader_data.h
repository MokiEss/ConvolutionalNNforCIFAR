//
// Created by Mokhtar on 28/07/2025.
//



#ifndef LOADER_DATA_H
#define LOADER_DATA_H

#include "utilities.h"
#include <filesystem>
#include <random>
namespace fs = std::filesystem;
Tensor load_image_as_tensor(const string& image_path);
vector<Tensor> load_image_as_tensors(string & base_dir, vector<int>& labels, vector<string>& image_paths);

#endif //LOADER_DATA_H
