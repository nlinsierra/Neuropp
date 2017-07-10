#include "MatrixOperations.h"

using namespace std;


// Оператор обращения к элементу матрицы
////////////////////////////////////////////////////////////////////////////////
double& Matrix2D::operator()(const int i, const int j) {
#ifdef POINTER_MATRIX
	return MatrixElements[(i - 1)*ColCount + j - 1];
#else
	return MatrixElements[i - 1][j - 1];
#endif
};
////////////////////////////////////////////////////////////////////////////////


// Оператор обращения к элементу матрицы
////////////////////////////////////////////////////////////////////////////////
double Matrix2D::operator()(const int i, const int j) const {
#ifdef POINTER_MATRIX
	return MatrixElements[(i - 1)*ColCount + j - 1];
#else
	return MatrixElements[i - 1][j - 1];
#endif
};
////////////////////////////////////////////////////////////////////////////////


// Оператор сложения
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator+(const Matrix2D& m2) {
	Matrix2D m(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(i, j) = (*this)(i, j) + m2(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор сложения с числом
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator+(const double& v) {
	Matrix2D m(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(i, j) = (*this)(i, j) + v;
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор вычитания
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator-(const Matrix2D& m2) {
	Matrix2D m(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(i, j) = (*this)(i, j) - m2(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор отрицания
////////////////////////////////////////////////////////////////////////////////
Matrix2D operator-(const Matrix2D& m1) {
	Matrix2D m(m1.GetRowCount(), m1.GetColCount());
	for (int i = 1; i <= m1.RowCount; i++) for (int j = 1; j <= m1.ColCount; j++)
		m(i, j) = -m1(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор умножения матриц
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator*(const Matrix2D& m2) {
	Matrix2D m;
	double sum = 0;
	int j, k;
	if (ColCount == m2.RowCount) {
		m = Matrix2D(RowCount, m2.ColCount);
#pragma omp parallel for private(k, j, sum)
		for (int i = 1; i <= RowCount; i++) for (k = 1; k <= m2.ColCount; k++) {
			sum = 0;
			for (j = 1; j <= ColCount; j++) sum += (*this)(i, j)*m2(j, k);
			m(i, k) = sum;
		}
	}
	else if (RowCount == m2.RowCount && ColCount == m2.ColCount) {
		m = Matrix2D(RowCount, ColCount);
		for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
			m(i, j) = (*this)(i, j)*m2(i, j);
	}
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор умножения матрицы на число
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator*(const double& v) {
	Matrix2D m(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(i, j) = v*(*this)(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор умножения числа на матрицу
////////////////////////////////////////////////////////////////////////////////
Matrix2D operator*(const double v, const Matrix2D& m2) {
	Matrix2D m(m2.RowCount, m2.ColCount);
	for (int i = 1; i <= m2.RowCount; i++) for (int j = 1; j <= m2.ColCount; j++)
		m(i, j) = v*m2(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////





// Оператор деления матриц
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator/(Matrix2D& m2) {
	Matrix2D m(RowCount, ColCount);
	m = (*this)*m2.Inverse();
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Оператор деления матрицы на число
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::operator/(const double& v) {
	Matrix2D m(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(i, j) = (*this)(i, j) / v;
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Вычисление обратной матрицы с помощью LUP-преобразования
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::Inverse() {
	Matrix2D m(RowCount, ColCount), P, C;
	C = LUP(P);
	for (int k = RowCount; k > 0; k--) {
		m(k, k) = 1;
		for (int j = RowCount; j > k; j--) m(k, k) -= C(k, j)*m(j, k);
		m(k, k) /= C(k, k);
		for (int i = k - 1; i > 0; i--) {
			for (int j = RowCount; j > i; j--) {
				m(i, k) -= C(i, j)*m(j, k);
				m(k, i) -= C(j, i)*m(k, j);
			}
			m(i, k) /= C(i, i);
		}
	}
	m = m*P;
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// LU-преобразование
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::LU() const {
	Matrix2D L(RowCount, ColCount), U(RowCount, ColCount), ResLU(RowCount, ColCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++) {
		U(1, i) = (*this)(1, i);
		if (U(1, 1) == 0) L(i, 1) = 0;
		else L(i, 1) = (*this)(i, 1) / U(1, 1);
		double sum = 0;
		for (int k = 1; k < i; k++) sum += L(i, k)*U(k, j);
		U(i, j) = (*this)(i, j) - sum;
		if (i > j) L(j, i) = 0;
		else {
			sum = 0;
			for (int k = 1; k < i; k++) sum += L(j, k)*U(k, i);
			if (U(i, i) == 0) L(j, i) = 0;
			else L(j, i) = ((*this)(j, i) - sum) / U(i, i);
		}
	}
	ResLU = L + U - CalculateOnes(RowCount);
	return ResLU;
}
////////////////////////////////////////////////////////////////////////////////




// Перемена i-й и j-й строк матрицы местами
////////////////////////////////////////////////////////////////////////////////
void Matrix2D::SwapRows(int i, int j) {
	for (int c = 1; c <= ColCount; c++) {
		double tmp = (*this)(i, c);
		(*this)(i, c) = (*this)(j, c);
		(*this)(j, c) = tmp;
	}
}
////////////////////////////////////////////////////////////////////////////////




// LUP-преобразование
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::LUP(Matrix2D &P) {
	Matrix2D C(RowCount, ColCount);
	C = *this;
	P = CalculateOnes(RowCount);
	for (int i = 1; i <= RowCount; i++) {
		double PVal = 0;
		int PInd = -1;
		for (int row = i; row <= RowCount; row++)
			if (fabs(C(row, i)) > PVal) { PVal = fabs(C(row, i)); PInd = row; }
		if (PVal == 0) return C;
		P.SwapRows(PInd, i); C.SwapRows(PInd, i);
		for (int j = i + 1; j <= RowCount; j++) {
			C(j, i) /= C(i, i);
			for (int k = i + 1; k <= RowCount; k++) C(j, k) -= C(j, i)*C(i, k);
		}
	}
	return C;
}
////////////////////////////////////////////////////////////////////////////////




// Вычисление детерминанта
////////////////////////////////////////////////////////////////////////////////
double Matrix2D::Det() {
	Matrix2D tmp;
	if (ColCount == 1) return (*this)(1, 1);
	tmp = LU();
	double prod = 1.0;
	for (int i = 1; i <= RowCount; i++) prod *= tmp(i, i);
	return prod;
}
////////////////////////////////////////////////////////////////////////////////





// Операция транспонирования
////////////////////////////////////////////////////////////////////////////////
Matrix2D Matrix2D::Transpose() {
	Matrix2D m(ColCount, RowCount);
	for (int i = 1; i <= RowCount; i++) for (int j = 1; j <= ColCount; j++)
		m(j, i) = (*this)(i, j);
	return move(m);
}
////////////////////////////////////////////////////////////////////////////////




// Нахождение следа матрицы
////////////////////////////////////////////////////////////////////////////////
double Matrix2D::Trace() {
	double res = 0;
	for (int i = 1; i <= RowCount; i++) res += (*this)(i, i);
	return res;
}
////////////////////////////////////////////////////////////////////////////////




// Отображение матрицы
////////////////////////////////////////////////////////////////////////////////
void Matrix2D::ShowElements() {
	for (int i = 1; i <= RowCount; i++) {
		for (int j = 1; j <= ColCount; j++) cout << (*this)(i, j) << " ";
		cout << endl;
	}
};
////////////////////////////////////////////////////////////////////////////////




// Формирование единичной матрицы
////////////////////////////////////////////////////////////////////////////////
Matrix2D CalculateOnes(int SizeNum) {
	Matrix2D Res(SizeNum, SizeNum);
	for (int i = 1; i <= SizeNum; i++) Res(i, i) = 1;
	return Res;
}
////////////////////////////////////////////////////////////////////////////////



// Перевод одномерного массива чисел в матрицу
////////////////////////////////////////////////////////////////////////////////
#ifdef POINTER_MATRIX
Matrix2D PointerToMatrix2D(double* p, int NumRows, int NumCols) {
	Matrix2D res(NumRows, NumCols);
	int count = 0;
	for (int i = 1; i <= NumRows; i++) for (int j = 1; j <= NumCols; j++)
		res(i, j) = p[count++];
	return move(res);
}
#else
Matrix2D VectorToMatrix2D(vector<double> p, int NumRows, int NumCols) {
	Matrix2D res(NumRows, NumCols);
	int count = 0;
	for (int i = 1; i <= NumRows; i++) for (int j = 1; j <= NumCols; j++)
		res(i, j) = p[count++];
	return move(res);
}
#endif
////////////////////////////////////////////////////////////////////////////////
