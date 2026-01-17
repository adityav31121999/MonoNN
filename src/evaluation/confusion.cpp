#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include "progress.hpp"

/**
 * @brief Calculates various classification metrics from a confusion matrix.
 * @param confusionMatrix A 2D vector representing the confusion matrix where
 *      confusionMatrix[i][j] is the number of observations in group i predicted
 *      to be in group j.
 * @return A confMat struct containing accuracy, precision, recall, F1-score, and 
 *      support for each class.
 */
confMat confusionMatrixFunc(const std::vector<std::vector<int>>& confusionMatrix) {
    confMat result;
    size_t n = confusionMatrix.size(); // number of classes
    if (n == 0) return result;

    result.precision.resize(n);
    result.recall.resize(n);
    result.f1.resize(n);
    result.support.resize(n);

    std::vector<int> truePositives(n, 0);
    std::vector<int> trueNegatives(n, 0);
    std::vector<int> predictedPositives(n, 0);
    std::vector<int> actualPositives(n, 0);

    int totalCorrect = 0;
    int totalSamples = 0;

    // Extract TP, row sums (actual), column sums (predicted)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int val = confusionMatrix[i][j];
            if (i == j) {
                truePositives[i] = val;     // true positives are diagonal elements (i == j)
                totalCorrect += val;        // all digonal elements
            }
            actualPositives[i] += val;        // row sum
            predictedPositives[j] += val;     // column sum
            totalSamples += val;
        }
        result.support[i] = actualPositives[i];
    }

    // Calculate overall accuracy
    result.avgAccuracy = totalSamples > 0 ? static_cast<double>(totalCorrect) / totalSamples : 0.0;
    result.accuracy.resize(n);

    // Calculate per-class precision, recall, f1
    double sum_f1_weighted = 0.0;
    double sum_f1_macro = 0.0;

    for (size_t i = 0; i < n; ++i) {
        // Per-class accuracy = (TP + TN) / (TP + TN + FP + FN)
        int falsePositives = predictedPositives[i] - truePositives[i];
        int falseNegatives = actualPositives[i] - truePositives[i];
        trueNegatives[i] = totalSamples - truePositives[i] - falsePositives - falseNegatives;

        result.accuracy[i] = totalSamples > 0
            ? static_cast<float>(truePositives[i] + trueNegatives[i]) / totalSamples
            : 0.0f;

        // Precision = TP / (TP + FP)
        float prec = (truePositives[i] + predictedPositives[i] - truePositives[i]) > 0
            ? static_cast<float>(truePositives[i]) / (truePositives[i] + predictedPositives[i] - truePositives[i])
            : 0.0f;

        // Recall = TP / (TP + FN)
        float rec = actualPositives[i] > 0
            ? static_cast<float>(truePositives[i]) / actualPositives[i]
            : 0.0f;

        // F1 = 2 * (precision * recall) / (precision + recall)
        float f1_score = (prec + rec > 0.0f)
            ? 2.0f * prec * rec / (prec + rec)
            : 0.0f;

        result.precision[i] = prec;
        result.recall[i] = rec;
        result.f1[i] = f1_score;

        sum_f1_macro += f1_score;
        sum_f1_weighted += f1_score * actualPositives[i];
    }

    result.macro_f1Score = n > 0 ? sum_f1_macro / n : 0.0;
    result.weighted_f1Score = totalSamples > 0 ? sum_f1_weighted / totalSamples : 0.0;

    return result;
}

/**
 * @brief Prints a formatted confusion matrix to the console.
 * @param confusionMatrix The confusion matrix to print.
 */
void printConfusionMatrix(const std::vector<std::vector<int>>& confusionMatrix) {
    size_t n = confusionMatrix.size();
    
    std::cout << "Confusion Matrix:\n";
    std::cout << std::setw(8) << "";
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setw(8) << "Pred " << i;
    }
    std::cout << std::setw(10) << "Total" << "\n";
    
    std::cout << std::string(10 + n * 8, '-') << "\n";
    
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setw(6) << "True " << i << " |";
        int rowSum = 0;
        for (size_t j = 0; j < n; ++j) {
            std::cout << std::setw(8) << confusionMatrix[i][j];
            rowSum += confusionMatrix[i][j];
        }
        std::cout << std::setw(8) << rowSum << "\n";
    }
    
    std::cout << std::string(10 + n * 8, '-') << "\n";
    
    std::cout << std::setw(8) << "Total";
    for (size_t j = 0; j < n; ++j) {
        int colSum = 0;
        for (size_t i = 0; i < n; ++i) {
            colSum += confusionMatrix[i][j];
        }
        std::cout << std::setw(8) << colSum;
    }
    std::cout << "\n\n";
}

/**
 * @brief Prints a detailed classification report including precision, recall, F1-score, and support for each class.
 * @param cm The confMat struct containing the calculated metrics.
 * @param classNames An optional vector of strings containing the names of the classes. If provided, they will be used in the report.
 */
void printClassificationReport(const confMat& cm, const std::vector<std::string>& classNames) {
    size_t n = cm.precision.size();
    bool hasNames = !classNames.empty() && classNames.size() == n;
    
    std::cout << "Classification Report:\n";
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << std::setw(12) << "Class"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall"
              << std::setw(12) << "F1-Score"
              << std::setw(12) << "Support\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < n; ++i) {
        std::string name = hasNames ? classNames[i] : "Class " + std::to_string(i);
        std::cout << std::setw(12) << name
                  << std::setw(12) << cm.precision[i]
                  << std::setw(12) << cm.recall[i]
                  << std::setw(12) << cm.f1[i]
                  << std::setw(12) << cm.support[i] << "\n";
    }
    
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(36) << "Avg. Accuracy" << std::setw(24) << cm.avgAccuracy << "\n";
    std::cout << std::setw(36) << "Macro Avg F1" << std::setw(24) << cm.macro_f1Score << "\n";
    std::cout << std::setw(36) << "Weighted Avg F1" << std::setw(24) << cm.weighted_f1Score << "\n\n";
}
