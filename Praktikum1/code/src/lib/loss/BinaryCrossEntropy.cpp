#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("Argument vectors are not of the same size.");
    }
    
    const std::size_t size = predictedOutput.size();
    
    std::vector<double> vecP;
    std::vector<double> vecQ;
    std::vector<double> vecR;
    
    vecP.reserve(size);
    vecQ.reserve(size);
    vecR.reserve(size);

    for (std::size_t i = 0; i < size; i += 1) {
        const double pred = predictedOutput[i];
        const double act = actualOutput[i];
        vecP.push_back(std::log(clampProbability(pred)));
        vecQ.push_back(1.0 - std::log(clampProbability(pred)));
        vecR.push_back(1.0 - act);
    }

    return -1.0 * (dotProduct(actualOutput, vecP) + dotProduct(vecR, vecQ)) / static_cast<double>(size); 
}

double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("Argument vectors are not of the same size.");
    }
    
    const std::size_t size = predictedOutput.size();
    std::size_t correct = 0;

    for (std::size_t i = 0; i < size; i += 1) {
        const double v_i = (predictedOutput[i] > 0.5) ? 1.0 : 0.0;
        if (v_i == actualOutput[i]) correct += 1;
    }
    
    return static_cast<double>(correct) / static_cast<double>(size);
}
 
std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("Argument vectors are not of the same size.");
    }
    
    const std::size_t size = predictedOutput.size();
    std::vector<double> vecGrad;
    vecGrad.reserve(size);

    for (std::size_t i = 0; i < size; i += 1) {
        const double act = actualOutput[i];
        const double clampedPred = clampProbability(predictedOutput[i]);
        const double numerator = act / clampedPred - ((1.0 - act) / (1.0 - clampedPred));
        vecGrad.push_back(numerator / static_cast<double>(size));
    }
    
    return vecGrad;
}

}
