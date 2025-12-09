#ifndef PROGRESS_HPP
#define PROGRESS_HPP 1
#include <vector>
#include <string>

#define LEARNING_MAX 0.001f         // maximum learning rate allowed
#define LEARNING_MIN 0.00001f       // minimum learning rate allowed
#define LAMBDA_L1 0.0001f           // L1 regularization parameter
#define LAMBDA_L2 0.0025f           // L2 regularization parameter
#define DROPOUT_RATE 0.50f          // dropout rate
#define DECAY_RATE 0.0025f          // weight decay rate
#define WEIGHT_DECAY 0.001f         // weight decay parameter
#define SOFTMAX_TEMP 1.05f          // softmax temperature
#define EPOCH 100                   // epochs for single set training
#define SESSION_SIZE 5              // number of batches in single session
#define BATCH_SIZE 16                // number of inputs in single batch
#define ALPHA 0.80f                 // gradient splitting factor

// struct to hold statistical information about data
struct Statistics {
    float mean;     // mean value
    float std;      // standard deviation
    float min;      // minimum value from the set
    float max;      // maximum value from the set
};

Statistics computeStats(const std::vector<float>& data);
Statistics computeStats(const std::vector<std::vector<float>>& data);
Statistics computeStats(const std::vector<std::vector<std::vector<float>>>& data);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<float>>& act);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<std::vector<float>>>& act);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<std::vector<std::vector<float>>>>& act);

// struct to save and access information on training of neural network
// single session will have fixed number of batches or files to be trained on
struct progress {
    unsigned int sessionSize;           // number of batches to be trained in single session (1 or many)
    unsigned int filesProcessed;        // number of files processed in training so far
    unsigned int batchSize;             // number of files in single batch (1 or many, for mini-batch)
    unsigned int totalTrainFiles;       // total training files
    unsigned int epoch;                         // for full data training epoch (for mini-batch and full dataset)
    float trainingPredictions;                  // correct training predictions in full dataset training
    float currentLearningRate;                  // current session's learning rate after successful training
    float loss;                                 // loss after successful training
    double accLoss;                             // accumulated loss till current session
    float trainAccuracy;                        // training accuracy in full data set training (for mini-batch and full dataset)
    unsigned long long totalCycleCount;         // total cycles after full training
    unsigned int totalSessionsOfTraining;       // total sessions used for training
    double timeForCurrentSession;               // time taken for current session
    double timeTakenForTraining;                // total time taken throughout sessions
};

// struct to save testing data
struct test_progress {
    // files
    unsigned int totalTestFiles;        // total test files
    unsigned int testFilesProcessed;    // number of files processed in testing so far

    // testing
    float testError;                    // error recorded during testing
    float testAccuracy;                 // accuracy recorded during testing
    unsigned int correctPredictions;    // correct predictions done in testing
};

bool logProgressToCSV(const progress& p, const std::string& filePath);
bool loadLastProgress(progress& p, const std::string& filePath);
bool logTestProgressToCSV(const test_progress& p, const std::string& filePath);
bool loadLastTestProgress(test_progress& p, const std::string& filePath);

#endif // PROGRESS_HPP