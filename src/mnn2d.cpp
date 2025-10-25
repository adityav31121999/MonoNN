#include "include/mnn.hpp"

/**
 * @brief Constructor for the mnn2d class.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param layers Number of hidden layers.
 * @param order Order of the monomial.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int outh, int layers, float order) :
    inWidth(inw), inHeight(inh), outWidth(outw), outHeight(outh), layers(layers),
    order(order), width(layers, 0), batchSize(1)
{
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // set hidden layers width and height
    for(int i = 0; i < layers - 1; i++) {
        width[i] = inw;
    }
    width[layers-2] = outw;
    outw = outh;

    // dot product and their activations
    dotProds[0].resize(inh, std::vector<float>(width[0], 0.0f));
    for (int i = 1; i < layers-2; i++) {
        dotProds[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        activate[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(width[0], 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(outw, std::vector<float>(outh));
    bweights[layers-1].resize(outw, std::vector<float>(outh));
    cgradients[layers-1].resize(outw, std::vector<float>(outh));
    bgradients[layers-1].resize(outw, std::vector<float>(outh));
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
mnn2d::mnn2d(int inw, int inh, int outw, int outh, int dim, int layers, float order) :
    order(order), inWidth(inw), inHeight(inh), outWidth(outw), outHeight(outh),
    width(layers, dim), batchSize(1), layers(layers)
{    
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // set hidden layers width and height
    // set hidden layers width and height
    for(int i = 0; i < layers - 1; i++) {
        width[i] = dim;
    }
    outw = outw;

    // dot product and their activations
    dotProds[0].resize(inh, std::vector<float>(width[0], 0.0f));
    activate[0].resize(inh, std::vector<float>(width[0], 0.0f));
    for (int i = 1; i < layers-1; i++) {
        dotProds[i].resize(width[i-1], std::vector<float>(width[i]));
        activate[i].resize(width[i-1], std::vector<float>(width[i]));
    }
    

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(outw, std::vector<float>(outh));
    bweights[layers-1].resize(outw, std::vector<float>(outh));
    cgradients[layers-1].resize(outw, std::vector<float>(outh));
    bgradients[layers-1].resize(outw, std::vector<float>(outh));
}

/**
 * @brief Constructor for the mnn2d class.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param dim Dimension of hidden layers.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int outh, std::vector<int> width, float order) :
    inWidth(inw), inHeight(inh), outWidth(outw), outHeight(outh),
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
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i-1], std::vector<float>(this->width[i]));
        activate[i].resize(width[i-1], std::vector<float>(this->width[i]));
    }

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(outw, std::vector<float>(outh));
    bweights[layers-1].resize(outw, std::vector<float>(outh));
    cgradients[layers-1].resize(outw, std::vector<float>(outh));
    bgradients[layers-1].resize(outw, std::vector<float>(outh));
}