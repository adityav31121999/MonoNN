#include "mnn.hpp"

/**
 * @brief Constructor for the mnn class.
 * @param insize Input size.
 * @param outsize Output size.
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int layers, float order) :
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1)
{
    // set width of hidden layers and dot products
    // int dim = (insize > outsize) ? insize : outsize;     // (optional)
    int dim = (insize + outsize) / 2;
    width.resize(layers, dim);
    width[layers - 1] = outsize;
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
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1)
{
    // set width of hidden layers and dot products
    width.resize(layers, dim);
    width[layers - 1] = outsize;
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
    for (int i = 1; i < layers-1; i++) {
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
    order(order), inSize(insize), outSize(outsize), width(width), layers(width.size()),
    input(insize, 0.0f), output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1)
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
    for (int i = 1; i < layers-1; i++) {
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