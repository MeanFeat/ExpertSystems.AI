#pragma once
#include "es_core_pch.h"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic > MatrixDynamic;
namespace Eigen {
	template<class Matrix>
	void write_binary(const char* filename, const Matrix& matrix) {
		std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
		typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
		out.write((char*)(&rows), sizeof(typename Matrix::Index));
		out.write((char*)(&cols), sizeof(typename Matrix::Index));
		out.write((char*)matrix.data(), rows*cols * sizeof(typename Matrix::Scalar));
		out.close();
	}
	template<class Matrix>
	void read_binary(const char* filename, Matrix& matrix) {
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		typename Matrix::Index rows = 0, cols = 0;
		in.read((char*)(&rows), sizeof(typename Matrix::Index));
		in.read((char*)(&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read((char*)(matrix.data()), rows*cols * sizeof(typename Matrix::Scalar));
		in.close();
	}
	template<class Matrix>
	void peek_shape_binary(const char* filename, Matrix& matrix) {
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		typename Matrix::Index rows = 0, cols = 0;
		in.read((char*)(&rows), sizeof(typename Matrix::Index));
		in.read((char*)(&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.close();
	}
	template<class Matrix>
	void write_csv(const char* filename, Matrix& matrix) {
		std::ofstream file;
		file.open(filename);
		for (int i = 0; i < matrix.rows() * matrix.cols(); ++i) {
			file << (*(matrix.data() + i));
			if ((i % matrix.rows()) == matrix.rows() - 1) {
				file << std::endl;
			}
			else {
				file << ',';
			}
		}
		file.close();
	}
	MatrixXf BuildMatFromFile(std::string fName);
	MatrixXf PeekMatrixShapeFromFile(std::string fName);
	const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
	void writeToCSVfile(const std::string& name, const MatrixXf& matrix);

	void removeColumn(MatrixXf& matrix, unsigned int colToRemove);
}
