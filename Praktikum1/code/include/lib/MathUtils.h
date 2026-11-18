#pragma once

#include <vector>
#include <algorithm>
namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);

template<typename T>
std::vector<double> normalize(std::vector<T>& values, int maxValue) {
    if (values.empty()) return {};
    const auto max = static_cast<double>(maxValue);

    std::vector<double> result;
    result.reserve(values.size());

    for (const T& value : values) {
        result.push_back(static_cast<double>(value) / max);
    }
    
    return result;
}

template<typename T>
std::vector<double> normalize(const std::vector<T>& values) {
    if (values.empty()) return {};
    const auto max = static_cast<double>(*std::max_element(values.begin(), values.end()));
    
    std::vector<double> result;
    result.reserve(values.size());

    for (const T& value : values) {
        result.push_back(static_cast<double>(value) / max);
    }
    
    return result;
}

// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

}
