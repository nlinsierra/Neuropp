#pragma once

#include <stdio.h>
#include <string.h>
#include "GeneticOperations.h"
#include <vector>
#include <string>
#include <time.h>
#include <functional>
#include <map>

using namespace std;

using TActivateFunction = function<double(double)>;

//Типы активационных функций
////////////////////////////////////////////////////////////////////////////////
#define TANH 0
#define SGMD 1
#define LINE 2
////////////////////////////////////////////////////////////////////////////////


const int RangeOut[2][3] = {
	{ -1, 0, -1 },
	{ 1, 1,  1 }
};
const int RangeIn[2][3] = {
	{ -2, -4, -1 },
	{ 2,  4,  1 }
};


extern bool StopTraining;

// Массив активационных функций
////////////////////////////////////////////////////////////////////////////////
const map<int, TActivateFunction> ActivateFunctions = {
	{ TANH, [](double x) { return tanh(x); } },
	{ SGMD, [](double x) { return ((x < -35) ? 1e-12 : 1.0 / (1.0 + exp(-x))); } },
	{ LINE, [](double x) { return x; } }
};

const map<int, TActivateFunction> ActivateFunctionDerivatives = {
	{ TANH, [](double x) { return 1.0 - tanh(x)*tanh(x); } },
	{ SGMD, [](double x) { return ActivateFunctions.at(SGMD)(x)*(1.0 - ActivateFunctions.at(SGMD)(x)); } },
	{ LINE, [](double x) { return 1.0; } }
};
////////////////////////////////////////////////////////////////////////////////




// Набор параметров для обучения нейронной сети
////////////////////////////////////////////////////////////////////////////////
typedef struct {
	double Rate;
	double Alpha;
	int NumEpochs;
	double Error;
	double MinGrad;
	int PopSize;
	double CrossoverP;
	double MutationP;
	double InversionP;
	int SelStrategy;
	int ParentsStrategy;
} TrainParams;
////////////////////////////////////////////////////////////////////////////////





//Базовый класс нейронов
////////////////////////////////////////////////////////////////////////////////
class Neuron {
private:
	mutable	int				ActivateFunction__;	// Активационная функция
	mutable	double			Axon__;				// Выход
	mutable	double			State__;			// Состояние
	mutable double			Bias__;				// Смещение
public:
	// Конструкторы
	Neuron(void) : State__(0.0), Bias__(0.0), ActivateFunction__(0), Axon__(0.0) {};
	Neuron(int a) : State__(0.0), Bias__(0.0), ActivateFunction__(a), Axon__(0.0) {};

	void					State(double s) { State__ = s; };
	void					State(double s) const { State__ = s; };
	void					Bias(double b) { Bias__ = b; };
	void					Bias(double b) const { Bias__ = b; };
	void					ActivateFunction(int a) { ActivateFunction__ = a; };
	void					ActivateFunction(int a) const { ActivateFunction__ = a; };
	
	double					CalculateAxon(void) { return Axon__ = ActivateFunctions.at(ActivateFunction__)(State__); };
	double					CalculateAxon(void) const { return Axon__ = ActivateFunctions.at(ActivateFunction__)(State__); };
	double					Derivative() { return ActivateFunctionDerivatives.at(ActivateFunction__)(State__); };
	double					Derivative() const { return ActivateFunctionDerivatives.at(ActivateFunction__)(State__); };
	
	int						ActivateFunction(void) { return ActivateFunction__; };
	double					Axon(void) { return Axon__; };
	double					Axon(void) const { return Axon__; };
	double					Bias(void) { return Bias__; };
	double					Bias(void) const { return Bias__; };
};
////////////////////////////////////////////////////////////////////////////////




// Класс слоев для многослойного перцептрона
////////////////////////////////////////////////////////////////////////////////
class Layer {
private:
	int						ActivateFunction__;		// Активационная функция слоя и число нейронов в предыдущем слое
	vector<Neuron>			Neurons__;				// Массив нейронов слоя
	Matrix2D				Weights__;				// Веса нейронов слоя
	Matrix2D				Biases__;				// Смещения нейронов слоя
public:
	// Конструкторы
	Layer(void) : ActivateFunction__(0) {}; // Конструктор
	Layer(int NumNeurons, int ActivateFunction) : ActivateFunction__(ActivateFunction), Neurons__(vector<Neuron>(NumNeurons, Neuron(ActivateFunction))) {};

	void					Biases(const Matrix2D &b);
	void					Weights(const Matrix2D &w) { Weights__ = w; };
	void					CalculateAxons(void);
	void					CalculateStates(const vector<Neuron> &PrevLayer);
	void					CalculateStates(Matrix2D Inp);
	// Получение значений параметров
	int						ActivateFunction(void) { return ActivateFunction__; };
	int						NumNeurons() { return Neurons__.size(); };
	const Matrix2D&			Biases() { return Biases__; }
	const Matrix2D&			Weights() { return Weights__; }
	const double&			Weights(int i, int j) { return Weights__(i, j); }
	const vector<Neuron>&	Neurons() { return Neurons__; }
	const Neuron&			Neurons(int idx) { return Neurons__[idx]; };
};
////////////////////////////////////////////////////////////////////////////////




// Класс сетей - многослойных перцептронов
////////////////////////////////////////////////////////////////////////////////
class Net {
protected:
	Matrix2D				TrainSet__;			// Обучающая выборка
	Matrix2D				WeightsBiases__;	// Вектор весов и смещений
	Matrix2D				InputsRanges__;		// Диапазоны значений для входов
	vector<Layer>			Layers__;			// Слои сети

	void					InitInputs(int NumInputs);
	void					InitLayers(int NumInputs, int NumLayers, int AFun[], int NumLN[]);
private:

	vector<Matrix2D>		CalculateDelta(Matrix2D SimOut, Matrix2D AimOut);
	Matrix2D				CalculateGradient(Matrix2D SimOut, Matrix2D AimOut);
	Matrix2D				SumGradient();
	double					CalculateError();
	double					GoldSection(double &MinVal, double &MaxVal, Matrix2D Direction);
	double					CalculateOutputError(int NumOut, int PairCount);
	Matrix2D				CalculateJacobian();

	void					WBNetToLayers();
	void					RandomWB();
	vector<double>			GetWeights();
	vector<double>			GetBiases();
	int						GetNumWeights();
	int						GetNumBiases();

public:
	Net(void) : Layers__(vector<Layer>(0)) {};
	Net(int NumInputs, int NumLayers, int LayerAFuns[], int LayerSizes[]);

	int						NumInputs() { return Layers__.front().NumNeurons(); }; // Число входов
	int						NumOutputs() { return Layers__.back().NumNeurons(); }; // Число входов
	int						NumLayers() { return Layers__.size() - 1; }; // Число слоев
	const Matrix2D&			TrainSet() { return TrainSet__; };
	double					TrainSet(int i, int j) { return TrainSet__(i, j); };
	const Matrix2D&			WeightsBiases() { return WeightsBiases__; };
	double					WeightsBiases(int i, int j) { return WeightsBiases__(i, j); };
	const Matrix2D&			InputsRanges() { return InputsRanges__; };
	double					InputsRanges(int i, int j) { return InputsRanges__(i, j); };
	const vector<Layer>&	Layers() { return Layers__; };
	const Layer&			Layers(int idx) { return Layers__[idx]; };

	void					InputsRanges(const Matrix2D &ir) { InputsRanges__ = ir; };
	void					TrainSet(const Matrix2D &ts) { TrainSet__ = ts; };
	void					WeightsBiases(const Matrix2D &wb) { WeightsBiases__ = wb; };

	void					Simulate(Matrix2D Inputs, Matrix2D &Outputs);

	int						GradTrainOnLine(TrainParams Params, vector<double> &Error);
	int						GradTrainOffLine(TrainParams Params, vector<double> &Error);
	int						FastGradTrainOnLine(TrainParams Params, vector<double> &Error);
	int						FastGradTrainOffLine(TrainParams Params, vector<double> &Error);
	int						FRConjugateGradTrain(TrainParams Params, vector<double> &Error);
	int						RPropTrain(TrainParams Params, vector<double> &Error);
	int						LMTrain(TrainParams Params, vector<double> &Error);
	int						BayesianReg(TrainParams Params, vector<double> &Error);
	int						GeneticTraining(TrainParams Params, vector<double> &Error);

};
////////////////////////////////////////////////////////////////////////////////


// Класс сетей Элмана
////////////////////////////////////////////////////////////////////////////////
class Elman : public Net {
private:
	int ContextNeuronCount;
	Matrix2D GetHiddenAxons() {
		auto HiddenNeurons = Layers__[NumLayers() - 1].Neurons();
		vector<double> HiddenAxons(HiddenNeurons.size());
		transform(HiddenNeurons.begin(), HiddenNeurons.end(), HiddenAxons.begin(), [](Neuron n) { return n.Axon(); });
		return Matrix2D(vector<vector<double>>(1, HiddenAxons)).Transpose();
	};
public:
	void InitInputs(int NumInputs) { Net::InitInputs(NumInputs + ContextNeuronCount); };
	void InitLayers(int NumInputs, int NumLayers, int AFun[], int NumLN[]);
	Matrix2D Simulate(Matrix2D Inputs, Matrix2D &Outputs);
	Matrix2D Simulate(Matrix2D Inputs, Matrix2D &Outputs, Matrix2D ContextInputs);
};
////////////////////////////////////////////////////////////////////////////////


int sign(double x);
double SignedRandomVal(double Val);
double RandomVal(double Val);

