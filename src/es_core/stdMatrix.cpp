#include "es_core_pch.h"
#include "stdMatrix.h"



Eigen::MatrixXf Eigen::BuildMatFromFile(std::string fName)
{
    std::vector<std::vector<float>> tempMat;
    std::ifstream file(fName);
    std::string line;
    float temp;
    while (file.good()) {
        std::vector<float> col;
        std::getline(file, line);
        std::stringstream iss(line);
        std::string val;
        while (iss.good()) {
            std::getline(iss, val, ',');
            std::stringstream convertor(val);
            convertor >> temp;
            col.push_back(temp);
        }
        tempMat.push_back(col);
    }
    MatrixXf outMat(tempMat.size(), tempMat[0].size());
    for (int i = 0; i < static_cast<int>(tempMat.size()); i++) {
        for (int j = 0; j < static_cast<int>(tempMat[i].size()); j++) {
            outMat(i, j) = tempMat[i][j];
        }
    }
    return outMat;
}

Eigen::MatrixXf Eigen::PeekMatrixShapeFromFile(std::string fName)
{
    int rows = 0, cols = 0;
    std::ifstream file(fName);
    std::string line;
    while (file.good()) {
        std::getline(file, line);
        std::stringstream iss(line);
        if (rows == 0) {
            std::string val;
            while (iss.good()) {
                std::getline(iss, val, ',');
                cols++;
            }
        }
        rows++;
    }
    return MatrixXf::Zero(rows, cols);
}

void Eigen::writeToCSVfile(const std::string& name, const MatrixXf& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

void Eigen::removeColumn(MatrixXf& matrix, const unsigned int colToRemove)
{
    const unsigned int numRows = static_cast<unsigned int>(matrix.rows());
    const unsigned int numCols = static_cast<unsigned int>(matrix.cols()) - 1;

    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.rightCols(numCols - colToRemove);

    matrix.conservativeResize(numRows, numCols);
}
