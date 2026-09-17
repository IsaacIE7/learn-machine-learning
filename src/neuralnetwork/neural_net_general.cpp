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

using namespace std;


struct Layer {
    Mat weights;
    Vec bias;

    int neurons;
    int weightAmnt;

    Layer(int neurons, int weightAmnt): // layer with n neurons and k weights, weights and biases init to 0
        weights(Mat(neurons, weightAmnt)), 
        bias(neurons, 0),
        neurons(neurons),
        weightAmnt(weightAmnt)
        {
            for (int j = 0; j < weightAmnt; j++) {
                for (int i = 0; i < neurons; i++) {
                    weights.entries[i][j] = randomWeight();
                }
            }

            for (double& b: bias.comps) {
                b = randomWeight();
            }
        }

    Layer(Mat weights, Vec bias): 
        weights(weights),
        bias(bias), 
        neurons(weights.rows),
        weightAmnt(weights.cols) {}

    pair<Vec, Vec> forward_lyr_single_sample(const Vec& inputs) {
        Vec z = (weights * inputs) + bias;
        Vec p = z.sigmoid_element_wise();
        return {z, p};
    }

    pair<Mat, Mat> forward_lyr_batch(const Mat& inputs) {
        Mat z = (inputs * weights.transpose()).add_vec_to_row(bias);
        Mat p = z.sigmoid_element_wise();

        return {z, p};
    }

    pair<Mat, Mat> forward_lyr_softmax(const Mat& inputs) {
        Mat z = (inputs * weights.transpose()).add_vec_to_row(bias);
        Mat p = z.softmax_element_wise();

        return {z, p};
    }

    Mat compute_weight_grad(const Mat& prev_activation, const Mat& D) { // pass in p - y if output layer
        return D.transpose() * prev_activation * (1.0 / prev_activation.rows);
    }

    //delta = partial L with respect to z times x
    Mat compute_delta(const Mat& weights_next,  const Mat& D_next, const Mat& crnt_activation) {
        return  (D_next * weights_next).multiply_element_wise(crnt_activation.multiply_element_wise((crnt_activation - 1) * -1));
    }

    Mat compute_delta_output_lyr(const Mat& p, const Mat& y) {
        return p - y;
    }

    Vec compute_bias_grad(const Mat& D) {
        return (D.transpose() * Vec(D.rows, 1)) * (1.0 / D.rows);
    }

    double randomWeight() {
        return -1.0 + ((double)rand() / RAND_MAX) * 2.0;
    }
};


struct NeuralNet {
    vector<int> layout;
    vector<Layer> layers; // doesnt include input layer, just hidden and output layers

    vector<Mat> activations; // includes input layer, hidden layers, and output layer
    vector<Mat> zValues;

    vector<Mat> gradWeights;
    vector<Vec> gradBiases;

    NeuralNet(vector<int> layout):
    layout(layout), 
    activations({}), 
    zValues({}),
    gradWeights({}),
    gradBiases({})
    {
        for (int i = 1; i < layout.size(); i++) {
            layers.push_back(Layer(layout[i], layout[i - 1]));
        }
    }


   Mat forward(const Mat& data) {
        activations.clear();
        zValues.clear();

        activations.push_back(data);
        
        auto current = layers[0].forward_lyr_batch(data); //get first hidden layer activation and z values
        zValues.push_back(current.first); // add  first hidden layer z predictions to zvals list
        activations.push_back(current.second); // add first hidden layer activations ot activation list

        for (int i = 1; i < layers.size(); i++) {
            if (i == layers.size() - 1) {
                current = layers[i].forward_lyr_softmax(activations[i]); 
                zValues.push_back(current.first);
                activations.push_back(current.second); 
            } else {
                current = layers[i].forward_lyr_batch(activations[i]); //layers doesnt include input layer, activations does
                zValues.push_back(current.first);
                activations.push_back(current.second); 
            }
            
        }
        return activations.back();
    }

    double loss(const Mat& y) { //binary cross entropy
        Mat p = activations.back();

        Mat m1 = y.multiply_element_wise(p.log_element_wise());
        Mat m2 = ((y - 1) * (-1.0)).multiply_element_wise((((p - 1) * (-1.0)).log_element_wise()));
        
        Mat res = (m1 + m2) * -1;

        return (res * (1.0 / (y.rows * y.cols))).sum_entries(); 
    }

    // //categorical cross entropy
    // double loss(const Mat& y) { 
    //     Mat p = activations.back();

    //     Mat m1 = y.multiply_element_wise(p.log_element_wise());
    
    //     return ((m1  * -1) * (1.0 / (y.rows))).sum_entries(); 
    // }

    void backprop(const Mat& y) {
        gradWeights.clear();
        gradBiases.clear();

        Mat delta = layers.back().compute_delta_output_lyr(activations.back(), y);

        gradWeights.push_back(layers.back().compute_weight_grad(activations[layers.size() - 1], delta));
        gradBiases.push_back(layers.back().compute_bias_grad(delta));

        Mat delta_next = delta;

        for (int i = layers.size() - 2; i >= 0; i--) {
            delta = layers[i].compute_delta(layers[i + 1].weights, delta_next, activations[i + 1]);
            Mat gradcurrent = layers[i].compute_weight_grad(activations[i], delta);
            Vec gradbias = layers[i].compute_bias_grad(delta);
            delta_next = delta;

            gradWeights.push_back(gradcurrent);
            gradBiases.push_back(gradbias);
        }
        reverse(gradWeights.begin(), gradWeights.end());
        reverse(gradBiases.begin(), gradBiases.end());
    }

    void update(double learning_rate) {
        for (int i = 0; i < gradWeights.size(); i++) {
            layers[i].weights = layers[i].weights - (gradWeights[i] * learning_rate);
            layers[i].bias = layers[i].bias - (gradBiases[i] * learning_rate);
        }
    }

    void train(const Mat& data, const Mat& y, int epochs, double learning_rate, double tolerance) {
        forward(data);
        double current_loss = loss(y);
        int i = 0;

        while (i < epochs && current_loss > tolerance) {
            forward(data);
            current_loss = loss(y);

            // if (i % 50 == 0) {
            // cout << "iteration " << i << " Loss: " << current_loss 
            // << " Accuracy: " << accuracy_10(data, y) * 100 << "% " << endl;
            // }

            backprop(y);
            update(learning_rate);
            i++;
        }
    }

    Vec predict_one_ex(const Mat& input) { // should be 1 x m only one sample
        return forward(input).to_vector(); // if input is 1 x m output mat should be m x 1
    }

    Vec predict(const Vec& input) { // a forward pass but takes vector instead
        return forward(input.to_matrix()).to_vector(); 
    }

    Mat predict_softargmax(const Mat& data) { // returns the prediction matrix
        Mat P = forward(data);
        Mat res(P.rows, P.cols);


        for (int i = 0; i < P.rows; i++) {
            double max = P.entries[i][0];
            int in = 0;
            for (int j = 0; j < P.cols; j++) {
                if (P.entries[i][j] > max) {
                    max = P.entries[i][j]; 
                    in = j;
                }
            }
            res.entries[i][in] = 1;
        }

        return res;
    }

         //only to be used with ONE sample
    int predict_softargmax_classify_num(const Mat& data) { //returns index of category or num for mnist classification(change for other applications)
        Mat P = forward(data);
        Mat res(P.rows, P.cols);
        int in = 0;

        for (int i = 0; i < P.rows; i++) {
            double max = P.entries[i][0];
            in = 0;
            for (int j = 0; j < P.cols; j++) {
                if (P.entries[i][j] > max) {
                    max = P.entries[i][j]; 
                    in = j;
                }
            }
            res.entries[i][in] = 1;
        }

        return in;
    }


    double accuracy(const Mat& data, const Mat& y) {
        Mat pred = predict_softargmax(data);

        if (pred.rows != y.rows || pred.cols != y.cols) {
            throw invalid_argument("accuracy dimension mismatch");
        }

        int matchingentries = 0;
        int imagecorrect = 0;
        int total = y.rows;

        for (int i = 0; i < pred.rows; i++) {
            matchingentries = 0;
            for (int j = 0; j < pred.cols; j++) {
                if (pred.entries[i][j] == y.entries[i][j]) {
                    matchingentries++;
                }
            }
            if (matchingentries == pred.cols) imagecorrect++;
        }

        return (double)imagecorrect / total;
    }

    


};





//UTILS

void write_trained_vals(const NeuralNet& N, string filepath) { //write trained values to json file

    ofstream f(filepath); 
    if (f.is_open()) { 
        pair<vector<Mat>, vector<Vec>> data; 
        for (size_t i = 0; i < N.layers.size(); i++) { 
            data.first.push_back(N.layers[i].weights); 
            data.second.push_back(N.layers[i].bias); 
        } 

        json output = data; 
        f << output.dump(4);//4 is pretty printing
        f.flush();

        f.close(); 
        std::cout << "Successfully wrote data to file." << std::endl; 
    } else { 
        std::cerr << "Error: Could not open the file path for writing." << std::endl; 
    } 
}


pair<vector<Mat>, vector<Vec>> parse_trained_vals(string filepath) { //parse json datasets into custom mat and vec
    ifstream f(filepath);
    json data = json::parse(f);
    

    //library feature that converts the data into what was written to the file
    pair<vector<Mat>, vector<Vec>> trained_vals = data.get<pair<vector<Mat>, vector<Vec>>>(); 

    return trained_vals;
}



//END OF UTILS

int main() {

}
