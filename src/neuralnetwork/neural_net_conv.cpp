#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include <fstream>
#include <json.hpp>
using json = nlohmann::json;

#include "../matvec/vec.h"
#include "../matvec/mat.h"
#include "../matvec/tensor.h"

using namespace std;


struct ConvLayer {
    Tensor kernels;
    int height;// of kernel
    int width;
    int channels; //still of kernel
    vector<double> biases;
    bool padding;

    ConvLayer(int numKernels, int height, int width, int channels, bool padding)
        : kernels({numKernels, channels, height, width}), height(height), width(width), channels(channels), padding(padding) {
        for (double& n: kernels.data) {
            n = -1.0 + ((double)rand() / RAND_MAX) * 2.0;
        }

        biases = vector<double>(numKernels);
        for (double& n: biases) {
            n = -1.0 + ((double)rand() / RAND_MAX) * 2.0;
        }
    }

    double convolveAt(const Tensor& input, int batch, int kernelNum, int startRow, int startCol) { //specify where exactly to output in the feature map
        double sum = 0;                                                             //output depth will be handled in the forward pass 
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channels; k++) {
                    if (i + startRow < 0 || j + startCol < 0 || i + startRow  > input.get_shape()[2] - 1 || j + startCol > input.get_shape()[3] - 1) {
                        continue;
                    }
                    sum += kernels.get({kernelNum, k, i, j}) * input.get({batch, k, i + startRow, j + startCol});
                }
            }
        }
        return sum + biases[kernelNum];
    }

    Tensor forward(const Tensor& input, int stride) {
        int p;
        !padding ? p = 0 : p = (kernels.get_shape()[2] - 1)/2;
        const auto& shape = input.get_shape();
        int outputH = (input.get_shape()[2] + 2 * p - kernels.get_shape()[2])/stride + 1;
        int outputW = (input.get_shape()[3] + 2 * p - kernels.get_shape()[3])/stride + 1;
        Tensor featureMap({shape[0], kernels.get_shape()[0], outputH, outputW});

        for (int b = 0; b < shape[0]; b++) {
            for (int i = 0; i < outputH; i++) {
                for (int j = 0; j < outputW; j++) {
                    for (int k = 0; k < kernels.get_shape()[0]; k++) {
                        featureMap.data[featureMap.get_index({b, k, i, j})] = convolveAt(input, b, k, i * stride - p, j * stride - p);
                    }
                }
            }
        }
        return featureMap;
    }


};


struct NeuralNetConv {
};





//UTILS

// void write_trained_vals_conv(const NeuralNetConv& N, string filepath) { //write trained values to json file

//     ofstream f(filepath); 
//     if (f.is_open()) { 
//         pair<vector<Mat>, vector<Vec>> data; 
//         for (size_t i = 0; i < N.layers.size(); i++) { 
//             data.first.push_back(N.layers[i].weights); 
//             data.second.push_back(N.layers[i].bias); 
//         } 

//         json output = data; 
//         f << output.dump(4);//4 is pretty printing
//         f.flush();

//         f.close(); 
//         std::cout << "Successfully wrote data to file." << std::endl; 
//     } else { 
//         std::cerr << "Error: Could not open the file path for writing." << std::endl; 
//     } 
// }


// pair<vector<Mat>, vector<Vec>> parse_trained_vals_conv(string filepath) { //parse json datasets into custom mat and vec
//     ifstream f(filepath);
//     json data = json::parse(f);
    

//     //library feature that converts the data into what was written to the file
//     pair<vector<Mat>, vector<Vec>> trained_vals = data.get<pair<vector<Mat>, vector<Vec>>>(); 

//     return trained_vals;
// }



//END OF UTILS

int main() {
    // Input: [1, 1, 5, 5]
    //
    //  1  2  3  4  5
    //  6  7  8  9 10
    // 11 12 13 14 15
    // 16 17 18 19 20
    // 21 22 23 24 25

    Tensor input({1, 1, 5, 5});

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            input.data[input.get_index({0, 0, i, j})]
                = i * 5 + j + 1;
        }
    }

    // 1 kernel, 3x3, 1 channel, NO padding
    ConvLayer conv(1, 3, 3, 1, false);

    // Kernel = all ones
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            conv.kernels.data[
                conv.kernels.get_index({0, 0, i, j})
            ] = 1.0;
        }
    }

    conv.biases[0] = 0.0;

    // STRIDE = 2
    Tensor output = conv.forward(input, 2);

    cout << "Output shape: ";

    for (int dim : output.get_shape()) {
        cout << dim << " ";
    }

    cout << "\n\nOutput:\n";

    for (int i = 0; i < output.get_shape()[2]; i++) {
        for (int j = 0; j < output.get_shape()[3]; j++) {
            cout << output.get({0, 0, i, j}) << "\t";
        }

        cout << "\n";
    }

    return 0;
}