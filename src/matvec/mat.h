#pragma once

#include <vector>
#include <utility>

#include <json.hpp>
using json = nlohmann::json;
using namespace std;

struct Vec;

struct Mat {
    vector<vector<double>> entries;
    int rows;
    int cols;
    pair<int, int> dim;
    
    Mat();

    Mat(int rows, int cols); //all entries 0 m x n matrix

    Mat(vector<vector<double>> entries); // create matrix with initialized values

    Vec mat_mul(const Vec& v) const;

    Vec operator*(const Vec& v) const;

    Mat operator+(Mat m) const;

    Mat add_vec_to_row(const Vec& v);

    Mat operator+(double c) const;

    Mat operator-(Mat m) const;

    Mat operator-(double c) const;

    Mat operator*(double c) const;

    Vec row_vec(int row) const;

    Vec col_vec(int col) const;

    Mat operator*(const Mat& m) const;

    Mat multiply_element_wise(const Mat& m) const;

    Mat sigmoid_element_wise() const;

    Mat softmax_element_wise() const;

    Mat log_element_wise() const;

    double sum_entries() const;

    Mat transpose () const; //lets compiler know that original Mat isnt edited

    Vec to_vector() const;
};

//util
Mat get_batch_slice(const Mat& original, int start_row, int batch_size);

void to_json(nlohmann::json& j, const Mat& m);

//review
void from_json(const nlohmann::json& j, Mat& m);
