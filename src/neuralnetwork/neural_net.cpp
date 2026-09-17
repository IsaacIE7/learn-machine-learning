#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "raylib.h"

#include "mnist/mnist_reader_less.hpp"

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

    Mat compute_delta(const Mat& weights_next,  const Mat& D_next, const Mat& crnt_activation) {
        return  (D_next * weights_next).multiply_element_wise(crnt_activation.multiply_element_wise((crnt_activation - 1) * -1));
    }

    Mat compute_D_output_lyr(const Mat& p, const Mat& y) {
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

        Mat D = layers.back().compute_D_output_lyr(activations.back(), y);

        gradWeights.push_back(layers.back().compute_weight_grad(activations[layers.size() - 1], D));
        gradBiases.push_back(layers.back().compute_bias_grad(D));

        Mat Dnext = D;

        for (int i = layers.size() - 2; i >= 0; i--) {
            D = layers[i].compute_delta(layers[i + 1].weights, Dnext, activations[i + 1]);
            Mat gradcurrent = layers[i].compute_weight_grad(activations[i], D);
            Vec gradbias = layers[i].compute_bias_grad(D);
            Dnext = D;

            gradWeights.push_back(gradcurrent);
            gradBiases.push_back(gradbias);
        }
        reverse(gradWeights.begin(), gradWeights.end());
        reverse(gradBiases.begin(), gradBiases.end());
    }

    // void backprop(const Mat& y) {
    //     gradWeights.clear();
    //     gradBiases.clear();

    //     gradWeights.resize(layers.size(), Mat(0, 0));
    //     gradBiases.resize(layers.size(), Vec(0));

    //     Mat D = layers.back().compute_D_output(activations.back(), y);

    //     gradWeights[layers.size() - 1] = (layers.back().gradient_w_lyr(activations[layers.size() - 1], D));
    //     gradBiases[layers.size() - 1] = (layers.back().gradient_b_lyr(D));

    //     Mat Dnext = D;

    //     for (int i = layers.size() - 2; i >= 0; i--) {
    //         D = layers[i].compute_D(layers[i + 1].weights, Dnext, activations[i + 1]);
    //         Mat gradcurrent = layers[i].gradient_w_lyr(activations[i], D);
    //         Vec gradbias = layers[i].gradient_b_lyr(D);
    //         Dnext = D;

    //         gradWeights[i] = gradcurrent;
    //         gradBiases[i] = gradbias;
    //     }
    // }

    // void update(double learning_rate) {
    //     for (int i = 0; i < gradWeights.size(); i++) { // gradients start with output layer and moves backwards
    //         int in = gradWeights.size() - 1 - i;       //layers start with first hidden
    //         layers[i].weights = layers[i].weights - (gradWeights[in] * learning_rate);
    //         layers[i].bias = layers[i].bias - (gradBiases[in] * learning_rate);
    //     }
    // }

    void update(double learning_rate) {
        for (int i = 0; i < gradWeights.size(); i++) {
            layers[i].weights = layers[i].weights - (gradWeights[i] * learning_rate);
            layers[i].bias = layers[i].bias - (gradBiases[i] * learning_rate);
        }
    }

    // void train(const Mat& data, const Mat& y, int epochs, double learning_rate, double tolerance, int batch_size) {
    //     forward(data);
    //     double current_loss = loss(y);
    //     int i = 0;

    //     while (i < epochs && current_loss) {
    //         for (int start_row = 0; start_row < data.rows - batch_size; start_row += batch_size) {
    //             Mat batch = get_batch_slice(data, start_row, batch_size);
    //             Mat ybatch = get_batch_slice(y, start_row, batch_size);

    //             forward(batch);
                

    //             backprop(ybatch);
    //             update(learning_rate);
    //         }  
    //         if (i % 100 == 0) {
    //             cout << "iteration " << i
    //              << " Loss: " << current_loss
    //              << " Accuracy: " << accuracy_10(data, y)
    //              << endl;
    //         }
    //         forward(data);
    //         current_loss = loss(y);
            
    //         i++;

    //     }
    // }

    void train(const Mat& data, const Mat& y, int epochs, double learning_rate, double tolerance) {
        forward(data);
        double current_loss = loss(y);
        int i = 0;

        while (i < epochs && current_loss > tolerance) {
            forward(data);
            current_loss = loss(y);

            if (i % 50 == 0) {
            cout << "iteration " << i << " Loss: " << current_loss 
            << " Accuracy: " << accuracy_10(data, y) * 100 << "% " << endl;
            }

            backprop(y);
            update(learning_rate);
            i++;
        }
    }

    Vec predict_one_ex(const Mat& input) { // should be 1 x m only one sample
        return forward(input).to_vector(); // if input is 1 x m output mat should be m x 1
    }

    Vec predict_one_ex_bin(const Mat& input) {
        Vec v = predict_one_ex(input); 
        Vec res(v.dim);
        
        for (int i = 0; i < input.rows; i++) {
            res.comps[i] = v.comps[i] >= 0.5 ? 1 : 0;  
        }
        return res; 
    }

    Vec predict(const Vec& input) { 
        return forward(input.to_matrix()).to_vector(); 
    }

    Mat predict_binary(const Mat& data) {
        Mat p = forward(data);
        Mat res(p.rows, p.cols);

        for (int i = 0; i < p.rows; i++) {
            for (int j = 0; j < p.cols; j++) {
                res.entries[i][j] = p.entries[i][j] >= 0.5 ? 1.0 : 0.0;
            }
        }

        return res;
    }

    Mat predict_softargmax(const Mat& data) {
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
    int predict_softargmax_classify_num(const Mat& data) {
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

    double accuracy_binary(const Mat& data, const Mat& y) {
        Mat pred = predict_binary(data);

        if (pred.rows != y.rows || pred.cols != y.cols) {
            throw invalid_argument("accuracy dimension mismatch");
        }

        int correct = 0;
        int total = pred.rows * pred.cols;

        for (int i = 0; i < pred.rows; i++) {
            for (int j = 0; j < pred.cols; j++) {
                if (pred.entries[i][j] == y.entries[i][j]) {
                    correct++;
                }
            }
        }

        return (double)correct / total;
    }

    double accuracy_10(const Mat& data, const Mat& y) {
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


// 0's and 1's only
//uint8_t is unsigned 8 bit integer 0-255
pair<Mat, Mat> convert_data_binaryclassification(const mnist::MNIST_dataset<uint8_t, uint8_t>& dataset, int num_samples) {
    if (dataset.training_labels.empty())  throw invalid_argument("MNIST failed to load. Check file paths and filenames.");

    Mat X(num_samples, 784);
    Mat Y(num_samples, 1);

    const auto& training_images = dataset.training_images;
    const auto& training_labels = dataset.training_labels;
    // auto test_images = dataset.test_images;
    // auto test_labels = dataset.test_labels;
    
    int count = 0;

    for (int i = 0; i < training_labels.size() && count < num_samples; i++) {
        if (training_labels[i] == 1 || training_labels[i] == 0) {
            for (int j = 0; j < training_images[0].size(); j++) {
                 X.entries[count][j] = training_images[i][j] / 255.0; // use x[count] because i gets skipped sometimes
            }

            if (training_labels[i] == 1) {
                Y.entries[count][0] = 1;
            } else  {
                Y.entries[count][0] = 0;
            }   
            
            count++;
        }     
    }

    return {X, Y};
}

//uint8_t is unsigned 8 bit integer 0-255
pair<Mat, Mat> convert_data(const mnist::MNIST_dataset<uint8_t, uint8_t>& dataset, int num_samples) {
    if (dataset.training_labels.empty())  throw invalid_argument("MNIST failed to load. Check file paths and filenames.");

    Mat X(num_samples, 784);
    Mat Y(num_samples, 10);

    const auto& training_images = dataset.training_images;
    const auto& training_labels = dataset.training_labels;
    // auto test_images = dataset.test_images;
    // auto test_labels = dataset.test_labels;
    
    int count = 0;

    for (int i = 0 ; i < num_samples && i < training_labels.size(); i++) {
        for (int j = 0; j < training_images[0].size(); j++) {
                 X.entries[i][j] = training_images[i][j] / 255.0; 
        }    

        // i is sample num/row and training_labels[i] is the index we want to store 1 to represent the digit, i from 0-9
        //called one-hot encoding
        Y.entries[i][training_labels[i]] = 1;  
    }

    return {X, Y};
}


void display(vector<int> layout) { 
    const int screenWidth = 1200; 
    const int screenHeight = 1000; 

    const int neuronsize = max(2, 20 - layout[0] / 4);

    const int clean_vertical = screenHeight - 200;
    const int clean_horizontal = screenWidth - 400;

    const int input_amnt = layout[0]; 

    
    
    InitWindow(screenWidth, screenHeight, "Neural Network"); 
    SetTargetFPS(60); 

    while (!WindowShouldClose()) { 
        vector<pair<double, double>> circle_coords;
        BeginDrawing(); 
        ClearBackground(GRAY); 
        
        DrawText("input neurons", 250, 100, 25, GREEN);

        for (int i = 0; i < layout.size(); i++) {
            int y_spacing = (int)(1.0 * clean_vertical / (layout[i] + 1));         
            int x_spacing = (1.0 * clean_horizontal / (layout.size()));         
            for (int j = 1; j <= layout[i]; j++) {   
                int xPos = 50 + x_spacing * (i + 1);
                int yPos = 150 + y_spacing * j;

                circle_coords.push_back({xPos, yPos});

                DrawCircle(xPos, yPos, neuronsize, BLACK); 
            } 
        } 

        for (int i = 0; i < layout[0]; i++) {
            for (int j = 0; j < layout[1]; j++) {
                auto circle1 = circle_coords[i];
                auto circle2 = circle_coords[j + layout[0]];

                double startx = circle1.first;
                double starty = circle1.second;
                double endx = circle2.first;
                double endy = circle2.second;
                DrawLine(startx, starty, endx, endy, BLACK);
            }
        }

        for (int i = 0; i < layout[1]; i++) {
            for (int j = 0; j < layout[2]; j++) {
                auto circle1 = circle_coords[i + layout[0]];
                auto circle2 = circle_coords[j + layout[0] + layout[1]];

                double startx = circle1.first;
                double starty = circle1.second;
                double endx = circle2.first;
                double endy = circle2.second;
                DrawLine(startx, starty, endx, endy, BLACK);
            }
        }


        // DrawText("FPS: 60", 20, 15, 20, GREEN); 
        EndDrawing(); 
    } 
    
    CloseWindow(); 
}

//UTILS

int predict_digit(NeuralNet& N, const Mat& input) {
    Mat probs = N.forward(input);

    int bestIndex = 0;
    double bestValue = probs.entries[0][0];

    for (int j = 1; j < probs.cols; j++) {
        if (probs.entries[0][j] > bestValue) {
            bestValue = probs.entries[0][j];
            bestIndex = j;
        }
    }

    return bestIndex;
}

void draw_on_canvas(vector<double>& canvas, int row, int col, int radius = 1) {
    for (int dr = -radius; dr <= radius; dr++) {
        for (int dc = -radius; dc <= radius; dc++) {
            int r = row + dr;
            int c = col + dc;

            if (r >= 0 && r < 28 && c >= 0 && c < 28) {
                double dist2 = dr * dr + dc * dc;

                // Instead of a hard 1.0, fade the intensity based on distance from the center brush point
                double intensity = 1.0 - (sqrt(dist2) / (radius + 1));
                if (intensity > canvas[r * 28 + c]) {
                    canvas[r * 28 + c] = intensity; // Smooth anti-aliased edge!
                }
            }
        }
    }
}

void center_canvas(vector<double>& canvas) {
    double centerx;
    double centery;

    double totalM = 0;
    double xsum = 0;
    double ysum = 0;


    for (double& d: canvas) {
        totalM += d;
    }

    if (totalM == 0) return;

    for (int r = 0; r < 28; r++) {
        double rowsum = 0;
        for (int c = 0; c < 28; c++) {
            rowsum += canvas[r * 28 + c] * c;
        }
        xsum += rowsum;
    }
    centerx = round(xsum / (totalM));

    for (int c = 0; c < 28; c++) {
        double colsum = 0;
        for (int r = 0; r < 28; r++) {
            colsum += canvas[r * 28 + c] * r;
        }
        ysum += colsum;
    }
    centery = round(ysum / (totalM));

    int xshift = 14 - centerx;
    int yshift = 14 - centery;

    vector<double> centered(784);

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            if (r + yshift < 28 && r + yshift >= 0 && c + xshift < 28 && c + xshift >= 0) {
                centered[(r + yshift) * 28 + c + xshift] = canvas[r * 28 + c];
            }
        }
    }

    canvas = centered;
}


Mat canvas_to_mat(const vector<double>& canvas) {
    Mat input(1, 784);

    for (int i = 0; i < 784; i++) {
        input.entries[0][i] = canvas[i];
    }

    return input;
}

void print_probs(NeuralNet& N, const Mat& input) {
    Mat probs = N.forward(input);

    for (int j = 0; j < probs.cols; j++) {
        cout << j << ": " << probs.entries[0][j] * 100 << "% " << endl;
    }
    cout << endl;
}

void draw_digit_window(NeuralNet& N) {
    const int screenWidth = 800;
    const int screenHeight = 700;

    const int canvasX = 120;
    const int canvasY = 80;
    const int cellSize = 20;
    const int gridSize = 28;
    const int canvasSize = gridSize * cellSize;

    std::vector<double> canvas(28 * 28, 0.0);

    int prediction = -1;

    InitWindow(screenWidth, screenHeight, "Draw a digit");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        bool insideCanvas =
            mouse.x >= canvasX &&
            mouse.x < canvasX + canvasSize &&
            mouse.y >= canvasY &&
            mouse.y < canvasY + canvasSize;

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && insideCanvas) {
            int col = (mouse.x - canvasX) / cellSize;
            int row = (mouse.y - canvasY) / cellSize;

            draw_on_canvas(canvas, row, col, 1);
        }

        if (IsKeyPressed(KEY_C)) {
            fill(canvas.begin(), canvas.end(), 0.0);
            prediction = -1;
        }

        if (IsKeyPressed(KEY_ENTER)) {
            center_canvas(canvas);
            Mat input = canvas_to_mat(canvas);
            prediction = predict_digit(N, input);
            print_probs(N, input);
        }

        BeginDrawing();
        ClearBackground(DARKGRAY);

        DrawText("Draw digit with mouse", 120, 25, 24, WHITE);
        DrawText("ENTER = classify, C = clear", 120, 50, 20, LIGHTGRAY);

        DrawText(
                "Classification:",
                120,
                canvasY + canvasSize + 30,
                32,
                GREEN
            );

        // Draw canvas background
        DrawRectangle(canvasX, canvasY, canvasSize, canvasSize, BLACK);

        // Draw the 28x28 pixels enlarged
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                double val = canvas[r * 28 + c];
                unsigned char shade = (unsigned char)(val * 255.0);

                Color color = { shade, shade, shade, 255 };

                DrawRectangle(
                    canvasX + c * cellSize,
                    canvasY + r * cellSize,
                    cellSize,
                    cellSize,
                    color
                );
            }
        }

        // Optional grid lines
        for (int i = 0; i <= 28; i++) {
            DrawLine(canvasX, canvasY + i * cellSize, canvasX + canvasSize, canvasY + i * cellSize, GRAY);
            DrawLine(canvasX + i * cellSize, canvasY, canvasX + i * cellSize, canvasY + canvasSize, GRAY);
        }

        if (prediction != -1) {
            DrawText(
                TextFormat("Classification: %d", prediction),
                120,
                canvasY + canvasSize + 30,
                32,
                GREEN
            );
        }

        EndDrawing();
    }

    CloseWindow();
}


void write_trained_vals(const NeuralNet& N, string filepath) { 

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


pair<vector<Mat>, vector<Vec>> parse_trained_vals(string filepath) {
    ifstream f(filepath);
    json data = json::parse(f);
    

    //library feature that converts the data into what was written to the file
    pair<vector<Mat>, vector<Vec>> trained_vals = data.get<pair<vector<Mat>, vector<Vec>>>(); 

    return trained_vals;
}



//END OF UTILS


// int main() {
//     try {

// //std::vector<double> mnist_flattened_1_255; 

//         auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

//         auto train = convert_data(dataset, 5000);
//         Mat X_train = train.first;
//         Mat y_train = train.second;

//         NeuralNet N({784, 64, 32, 10});

//         cout << "Before training:" << endl;
//         N.forward(X_train);
//         cout << "Loss: " << N.loss(y_train) << endl;
//         cout << "Accuracy: " << N.accuracy_10(X_train, y_train) << endl;

//         cout << "\nTraining..." << endl;
//         //N.train(X_train, y_train, 1000, 0.1, 0.15);

//         cout << "\nAfter training:" << endl;
//         N.forward(X_train);
//         cout << "Loss: " << N.loss(y_train) << endl;
//         cout << "Accuracy: " << N.accuracy_10(X_train, y_train) << endl;

        
//         // Mat X(1, mnist_flattened_1_255.size()); 
//         // X.entries = {{mnist_flattened_1_255}};

//         // cout << "data set rows: " << X.rows << " cols: " << X.cols << endl;
//         // auto p = N.forward(X);
//         // cout << "data set rows: " << p.rows << " cols: " << p.cols << endl << endl;

//         // Mat P = N.predict_softargmax(X);

//         // Mat P2 = N.forward(X);

//         // for (auto d: P.entries) {
//         //     for (double m: d) {
//         //         cout << m << " ";
//         //     }
//         // }

//         // cout << endl << endl;

//         // int i = 0;
//         // for (auto d: P2.entries) {
//         //     for (double m: d) {
//         //         cout << "'" << i << "': " <<  m * 100 << "% ";
//         //     }
//         // }

//         // cout << endl << endl << "Digit classification: " << N.predict_softargmax_classify_num(X);

//     }
//     catch (const exception& e) {
//         cout << "Runtime error: " << e.what() << endl;
//     }

//     return 0;
// }

int main() {
    try {
        srand(0);
    
        NeuralNet N({784, 64, 10});
        
        
        auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

       

        auto data = convert_data(dataset, 10000);

        Mat X_data = data.first;
        Mat y_data = data.second;

        auto p = parse_trained_vals("C:/codingstuff/learnML/weights.json");
        int i = 0;
        for (auto& a: N.layers) {
            a.weights = p.first[i];
            a.bias = p.second[i];
            i++;
        }

        //N.train(X_data, y_data, 200, 3, 0.01);

        cout << "Train accuracy: " << N.accuracy_10(X_data, y_data) * 100 << "%" << endl;

        //write_trained_vals(N, "C:/codingstuff/learnML/weights.json");

        

        draw_digit_window(N);
    }
    catch (const exception& e) {
        cout << "Runtime error: " << e.what() << endl;
    }

    return 0;
}

// int main() {
//     display({6, 4, 2});
//     return 0;
// }

// int main() {
//     // auto dataset = mnist::read_dataset<uint8_t, uint8_t>();
//     // auto data = convert_data(dataset, 200);
//     // Mat X = data.first;
//     // Mat Y = data.second;

//     vector<vector<double>> entries = {{2, 1, 10, 6}};
//     Mat X = entries;
//     X = X.softmax_element_wise();
//     for (auto d: X.entries) {
//         for (double n: d) {
//             cout << n << " ";
//         }
//     }



//     return 0;
// }



