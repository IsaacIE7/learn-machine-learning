#include <iostream>
#include "vec.h"
#include "mat.h"
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

// {channels, height, width}
// {batch, channels, height, width}
    vector<int> shape;
    vector<int> strides;
    vector<double> data;
    //batch {5,3,2,4}
    // {3, 2, 4}
    // {chn * 2 * 4, row * 4 , 1 * col}
    Tensor::Tensor(initializer_list<int> init): shape(init), strides(shape.size(), 1){
        int n = 1;
        strides[shape.size() - 1] = n;
        for (int i = shape.size() - 2; i >= 0; i--) {
            n *= shape[i + 1];
            cout << " n = " << n << " ";
            strides[i] = n;
        }
        int m = 1;
        for (int k: shape) {
            m *=k ;
        }
        data = vector<double>(m, 0.0);
    }

    Tensor::Tensor(vector<int> shape, vector<double> data): shape(shape), strides(shape.size(), 1), data(data){
        int k = 1; 
        for (int n: shape) {
            k *= n;
        }
        if (k != data.size()) throw invalid_argument("Shape does not match data size");

        int n = 1;
        strides[shape.size() - 1] = n;
        for (int i = shape.size() - 2; i >= 0; i--) {
            n *= shape[i + 1];
            cout << " n = " << n << " ";
            strides[i] = n;
        }
    }

    int Tensor::get_index(vector<int> indices) const {
        int index = 0;
        for (int i = 0; i < indices.size(); i++) {
            index += indices[i] * strides[i];
        }
    }

    double Tensor::get(vector<int> indices) const {
        return data[get_index(indices)];
    }

    void Tensor::set(vector<int> indices, double val) {
        data[get_index(indices)] = val;
    }

    vector<int> Tensor::get_shape() const {
        return shape;
    }

    int Tensor::get_numel() {
        return data.size();
    }

    //operations
    Tensor Tensor::add(const Tensor& B) const {
        if (data.size() != B.data.size()) throw invalid_argument("Tensor addition size mismatch");
        vector<double> res(data.size());
        for (int i = 0; i < data.size(); i++) {
            res[i] = data[i] + B.data[i];
        }
        return Tensor(shape, res);
    }

    Tensor Tensor::operator+(const Tensor& B) const {
        return this->add(B);
    }

    Tensor Tensor::sub(const Tensor& B) const {
        if (data.size() != B.data.size()) throw invalid_argument("Tensor addition size mismatch");
        vector<double> res(data.size());
        for (int i = 0; i < data.size(); i++) {
            res[i] = data[i] - B.data[i];
        }
        return Tensor(shape, res);
    }

    Tensor Tensor::operator-(const Tensor& B) const {
        return this->sub(B);
    }

    Tensor Tensor::mul_elementwise(const Tensor& B) const {
        if (data.size() != B.data.size()) throw invalid_argument("Tensor addition size mismatch");
        vector<double> res(data.size());
        for (int i = 0; i < data.size(); i++) {
            res[i] = data[i] * B.data[i];
        }
        return Tensor(shape, res);
    }

    Tensor Tensor::operator*(const Tensor& B) const {
        return this->mul_elementwise(B);
    }

    Tensor Tensor::scale(double c) const {
        vector<double> res(data.size());
        for (int i = 0; i < data.size(); i++) {
            res[i] = data[i] * c;
        }
        return Tensor(shape, res);
    }

    Tensor Tensor::operator*(double c) const {
        return this->scale(c);
    }