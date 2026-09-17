#pragma once

#include <iostream>
#include "vec.h"
#include "mat.h"
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

// {channels, height, width}
// {batch, channels, height, width}
struct Tensor {
    vector<int> shape;
    vector<int> strides;
    vector<double> data;
    //batch {5,3,2,4}
    // {3, 2, 4}
    // {chn * 2 * 4, row * 4 , 1 * col}
    Tensor(initializer_list<int> init);

    Tensor(vector<int> shape, vector<double> data);;

    int get_index(vector<int> indices) const;

    double get(vector<int> indices) const;

    void set(vector<int> indices, double val);

    vector<int> get_shape() const;

    int get_numel();

    //operations
    Tensor add(const Tensor& B) const;

    Tensor operator+(const Tensor& B) const;

    Tensor sub(const Tensor& B) const;

    Tensor operator-(const Tensor& B) const;

    Tensor mul_elementwise(const Tensor& B) const;

    Tensor operator*(const Tensor& B) const;

    Tensor scale(double c) const;

    Tensor operator*(double c) const;
};