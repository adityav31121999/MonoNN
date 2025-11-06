#ifdef USE_CPU
#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>
#include <mutex>
#include "mnn.hpp"

/**
 * @brief train with using threads
 * @param in input vector
 * @param exp expected output vector
 */
void mnn::thredTrain(const std::vector<float>& in, const std::vector<float>& exp)
{
    // forprop

    // error

    // backprop

}


/**
 * @brief train on batch with using threads
 * @param in input vector
 * @param exp expected output vector
 */
void mnn::threadTrainBatch(const std::vector<std::vector<float>>& in, const std::vector<std::vector<float>>& exp)
{
    // forprop

    // error

    // backprop

}

#endif