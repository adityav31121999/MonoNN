#include "include/mnn.hpp"

/**
 * @brief Constructor for the mnn class.
 * @param insize Input size.
 * @param outsize Output size.
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int layers, float order) :
    order(order), inSize(insize), outSize(outsize), width(layers, insize),
    layers(layers), input(insize), output(outsize), target(outsize), batchSize(1),
    in2d(1, std::vector<float>(insize)), out2d(1, std::vector<float>(outsize)),
    tar2d(1, std::vector<float>(outsize))
{
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
}


/**
 * @brief Constructor for the mnn class.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int dim, int layers, float order) :
    order(order), inSize(insize), outSize(outsize), width(layers, dim),
    layers(layers), input(insize), output(outsize), target(outsize), batchSize(1),
    in2d(1, std::vector<float>(insize)), out2d(1, std::vector<float>(outsize)),
    tar2d(1, std::vector<float>(outsize))
{
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
}


/**
 * @brief Constructor for the mnn class.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, std::vector<int> width, float order) : 
    order(order), inSize(insize), outSize(outsize), width(width),
    input(insize), output(outsize), target(outsize), batchSize(1),
    in2d(1, std::vector<float>(insize)), out2d(1, std::vector<float>(outsize)),
    tar2d(1, std::vector<float>(outsize))
{
    layers = width.size();
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
}