#include "mnn.hpp"

/**
 * @brief Constructor for the mnn2d class.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param layers Number of hidden layers.
 * @param order Order of the monomial.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int layers, float order) :
    inWidth(inw), inHeight(inh), outWidth(outw), layers(layers),
    order(order), width(layers, 0), batchSize(1)
{
    // set hidden layers width and height
    int dim = (inw + outw) / 2;
    width.resize(layers, dim);
    width[layers - 1] = outw;

    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
}

/**
 * @brief Constructor for the mnn2d class.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param dim Dimension of hidden layers.
 * @param layers Number of hidden layers.
 * @param order Order of the monomial.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int dim, int layers, float order) :
    order(order), inWidth(inw), inHeight(inh), outWidth(outw),
    width(layers, dim), batchSize(1), layers(layers)
{
    // set hidden layers width and height
    width.resize(layers, dim);
    width[layers - 1] = outw;

    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
}

/**
 * @brief Constructor for the mnn2d class.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param dim Dimension of hidden layers.
 */
mnn2d::mnn2d(int inw, int inh, int outw, std::vector<int> width, float order) :
    inWidth(inw), inHeight(inh), outWidth(outw),
    width(width), batchSize(1), layers(width.size())
{
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
}