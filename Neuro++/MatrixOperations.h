#pragma once

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <omp.h>

using namespace std;

// Класс для работы с двумерными матрицами
////////////////////////////////////////////////////////////////////////////////
class Matrix2D {
  private:
    int						RowCount;			// Число строк
    int						ColCount;			// Число столбцов
    vector<vector<double>>	MatrixElements;		// Массив элементов матрицы
	
  public:
    // Конструктор
    Matrix2D(void) : RowCount(0), ColCount(0) {};
	Matrix2D(int r, int c) : RowCount(r), ColCount(c), MatrixElements(vector<vector<double>>(r, vector<double>(c, 0.0))) {};
	Matrix2D(const vector<vector<double>> &v) : MatrixElements(v) {};

    int						GetRowCount() const { return RowCount; }; // Число строк
    int						GetColCount() const { return ColCount; }; // Число столбцов
	vector<vector<double>>	GetVector() { return MatrixElements; };

    double&					operator()(const int i, const int j); // Значение элемента матрицы
	double					operator()(const int i, const int j) const; // Значение элемента матрицы

    Matrix2D				operator+(const Matrix2D& m2); // Оператор сложения
    Matrix2D				operator+(const double& v); // Оператор сложения с числом
    Matrix2D				operator-(const Matrix2D& m2); // Оператор вычитания
    friend Matrix2D			operator-(const Matrix2D& m1); // Оператор отрицаниия
    Matrix2D				operator*(const Matrix2D& m2); // Оператор умножения матриц
    friend Matrix2D			operator*(const double v, const Matrix2D& m2); // Оператор умножения матрицы на число
    Matrix2D				operator*(const double& v); // Оператор умножения матрицы на число
    Matrix2D				operator/(Matrix2D& m2); // Оператор деления матриц
    Matrix2D				operator/(const double& v); // Оператор деления матрицы на число

    double					Det(); // Вычисление детерминаннта
    Matrix2D				Inverse(); // Вычисление обратной матрицы
    Matrix2D				Transpose(); // Вычисление транспонированной матрицы
    double					Trace(); // Нахождение следа матрицы
    Matrix2D				LU() const; // Нахождение LU-разложения матрицы
    Matrix2D				LUP(Matrix2D &P); // Нахождение LUP-разложения матрицы
    void					SwapRows(int i, int j); // Перемена i-й и j-й строк матрицы местами 
    
    // Перевод двумерной матрицы в одномерный массив с порядком хранения
    // данных по строкам
    vector<double> Matrix2DToVector() {
		vector<double> LineMatrixElements;
		for_each(MatrixElements.begin(), MatrixElements.end(), [&LineMatrixElements](vector<double> CurRow) {
			copy(CurRow.begin(), CurRow.end(), back_inserter(LineMatrixElements));
		});
		return LineMatrixElements;
	};
    void					ShowElements();
};
////////////////////////////////////////////////////////////////////////////////




// Операции для работы над матрицами
////////////////////////////////////////////////////////////////////////////////
Matrix2D CalculateOnes(int SizeNum);
Matrix2D VectorToMatrix2D(vector<double> p, int NumRows, int NumCols);
////////////////////////////////////////////////////////////////////////////////


