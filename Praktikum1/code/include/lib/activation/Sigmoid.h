#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

class Sigmoid : public ActivationFunction {
public:
    double activation(double x) {};

public:
    double activationBackwards(double x) {};

public:
    std::string getName() {};
};

}