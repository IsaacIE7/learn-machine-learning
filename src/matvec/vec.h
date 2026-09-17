#pragma once

#include <vector>
#include <initializer_list>
#include <json.hpp>

using namespace std;

struct Mat; // forward declaration Mat so Vec can reference it before Mat is defined

struct Vec {
    vector<double> comps;
    int dim = 0;

    Vec();

    Vec(int dim); //create n dimensional vector with components initialized to 0

    Vec(int dim, double val); //create n dimensional vector with components initialized to desired val

    Vec(vector<double> comps); //initialize with components
    
    //initializer list is fixed size, read only, and used mainly for constuctors and function arguments
    Vec(initializer_list<double> init); // allows vector declaration, Vec v = {1.0, 2.0, 3.0};

    //read and modify
    double& operator[](size_t index); //size_t is an unsigned 64 bit integer long long
    
    //read only
    // const at end means this function will not change the vector
    // const at start means you can see the number but cant edit it
    const double& operator[](size_t index) const;

    Vec operator+(const Vec& v);

    Vec operator-(const Vec& v);

    Vec operator*(double c) const;

    double dot(const Vec& v) const;

    double operator*(const Vec& v) const;

    // Declare Mat-returning Vec methods here. Their definitions come after Mat is fully defined.
    Mat transpose () const;
    Mat to_matrix() const;

    Vec log_element_wise() const;

    Vec add_element_wise(double x) const;

    Vec mul_element_wise(const Vec& v) const;

    Vec sigmoid_element_wise() const;
};

void to_json(nlohmann::json& j, const Vec& v);

void from_json(const nlohmann::json& j, Vec& v);
