#include "operators.hpp"
#include <cmath>

#define PI 3.14159265358979323846f  // value of pi

/**
 * @brief Cosine Annealing Learning Rate Scheduler. Implements the popular cosine annealing schedule (as used in SGDR).
 * The learning rate starts at MAX_LR and smoothly decreases to MIN_LR following the cosine curve over totalEpochs.
 * Formula: lr = MIN_LR + (MAX_LR - MIN_LR) × 0.5 × (1 + cos(pi × epoch / totalEpochs))
 * @param MAX_LR Initial (maximum) learning rate. Typical range: 1e-4 to 1.0 (commonly 0.01 – 0.3)
 * @param MIN_LR Minium Learning rate possible
 * @param epoch Current training epoch (0-based). Valid range: >= 0 (negative values are clamped to 0)
 * @param totalEpochs    Total number of epochs in one annealing cycle. Valid range: > 0 (recommended >= 10)
 * @return Annealed learning rate for the current epoch. Range: [MIN_LR, MAX_LR]
 */
float cosineAnnealing(float MAX_LR, float MIN_LR, int epoch, int totalEpochs)
{
    // Input validation and clamping
    if (totalEpochs <= 0) totalEpochs = 1;
    if (epoch < 0) epoch = 0;
    if (epoch >= totalEpochs) epoch = totalEpochs - 1;
    float progress = static_cast<float>(epoch) / static_cast<float>(totalEpochs);
    float cosine = 0.5f * (1.0f + std::cos(PI * progress));

    return MIN_LR + (MAX_LR - MIN_LR) * cosine;
}

/**
 * @brief Reduce Learning Rate on Plateau. Monitors validation loss and reduces the learning rate by a factor
 * when no improvement is observed for 'patience' consecutive epochs.
 * @param currentLR Current learning rate (will be reduced if plateau detected). Typical range: 1e-8 to 1.0
 * @param previousLoss Validation loss from the previous epoch. Should be a positive float (lower = better)
 * @param currentLoss Validation loss from the current epoch Should be a positive float
 * @param patienceCounter Reference to counter tracking epochs without improvement. Initialize to 0 before first call.
 *  Automatically managed by the function
 * @param patience Number of epochs to wait before reducing LR. Typical values: 3 to 20 (commonly 5–10)
 * @param factor Multiplicative factor to reduce LR (0.1 = 10× reduction). Typical values: 0.1 to 0.5 (0.1 is most common)
 * @return New (possibly reduced) learning rate. Guaranteed >= 1e-8 (minimum LR floor)
 */
float learningRateOnPlateau(float currentLR, float previousLoss, float currentLoss,
                            int& patienceCounter, int patience, float factor = 0.1f)
{
    const float improvementThreshold = 1e-4f;  // Minimum relative improvement to count as better
    const float minLR = 1e-8f;                 // Hard floor to prevent vanishing LR

    // Check for meaningful improvement
    if (currentLoss < previousLoss - improvementThreshold)
    {
        patienceCounter = 0;  // Reset counter on improvement
    }
    else
    {
        ++patienceCounter;

        if (patienceCounter >= patience)
        {
            currentLR *= factor;
            patienceCounter = 0;  // Reset counter after reduction (standard practice)

            if (currentLR < minLR)
                currentLR = minLR;
        }
    }

    return currentLR;
}