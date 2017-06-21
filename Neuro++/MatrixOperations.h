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

// ����� ��� ������ � ���������� ���������
////////////////////////////////////////////////////////////////////////////////
class Matrix2D {
  private:
    int						RowCount;			// ����� �����
    int						ColCount;			// ����� ��������
    vector<vector<double>>	MatrixElements;		// ������ ��������� �������
	
  public:
    // �����������
    Matrix2D(void) : RowCount(0), ColCount(0) {};
	Matrix2D(int r, int c) : RowCount(r), ColCount(c), MatrixElements(vector<vector<double>>(r, vector<double>(c, 0.0))) {};
	Matrix2D(const vector<vector<double>> &v) : MatrixElements(v) {};

    int						GetRowCount() const { return RowCount; }; // ����� �����
    int						GetColCount() const { return ColCount; }; // ����� ��������
	vector<vector<double>>	GetVector() { return MatrixElements; };

    double&					operator()(const int i, const int j); // �������� �������� �������
	double					operator()(const int i, const int j) const; // �������� �������� �������

    Matrix2D				operator+(const Matrix2D& m2); // �������� ��������
    Matrix2D				operator+(const double& v); // �������� �������� � ������
    Matrix2D				operator-(const Matrix2D& m2); // �������� ���������
    friend Matrix2D			operator-(const Matrix2D& m1); // �������� ����������
    Matrix2D				operator*(const Matrix2D& m2); // �������� ��������� ������
    friend Matrix2D			operator*(const double v, const Matrix2D& m2); // �������� ��������� ������� �� �����
    Matrix2D				operator*(const double& v); // �������� ��������� ������� �� �����
    Matrix2D				operator/(Matrix2D& m2); // �������� ������� ������
    Matrix2D				operator/(const double& v); // �������� ������� ������� �� �����

    double					Det(); // ���������� �������������
    Matrix2D				Inverse(); // ���������� �������� �������
    Matrix2D				Transpose(); // ���������� ����������������� �������
    double					Trace(); // ���������� ����� �������
    Matrix2D				LU() const; // ���������� LU-���������� �������
    Matrix2D				LUP(Matrix2D &P); // ���������� LUP-���������� �������
    void					SwapRows(int i, int j); // �������� i-� � j-� ����� ������� ������� 
    
    // ������� ��������� ������� � ���������� ������ � �������� ��������
    // ������ �� �������
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




// �������� ��� ������ ��� ���������
////////////////////////////////////////////////////////////////////////////////
Matrix2D CalculateOnes(int SizeNum);
Matrix2D VectorToMatrix2D(vector<double> p, int NumRows, int NumCols);
////////////////////////////////////////////////////////////////////////////////


