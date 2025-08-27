#include "loader_data.h"



int main() {
    vector<int> train_labels ;
    vector<int> test_labels ;
    string path = "../CIFAR-10-images-master/train" ;
    vector<string> path_names ;
    vector<Tensor> train_images = load_image_as_tensors(path, train_labels,path_names);
    PerformConvNN(train_images, train_labels,path_names) ;

    path_names.clear();
    path = "../CIFAR-10-images-master/test" ;
    vector<Tensor> test_images = load_image_as_tensors(path, test_labels,path_names);
    Evaluate_model(test_images, test_labels) ;
}
