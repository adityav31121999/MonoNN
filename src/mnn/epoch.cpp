#include <filesystem>
#include <fstream>
#include <iostream>
#include "progress.hpp"

void epochDataToCsv(const std::string& dataSetName, const int epoch, const std::vector<std::vector<float>>& weightStats,
                    const std::vector<std::vector<float>>& confusion, const confMat& scores)
{
    // log dataset name first (1st line)
    // log epoch (2nd epoch)
    // log stats
    // log matrix
    // log scores
}