#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {
// Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
class ScaledSigmoid : public ActivationFunction {
public:
    double activation(double x) {};

public:
    double activationBackwards(double x) {};

public:
    std::string getName() {};
    
public:
    ScaledSigmoid(double min, double max);

    double getMin() const;
    double getMax() const;

private:
    double min;
    double max;
};

}