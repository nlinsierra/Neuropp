#include "NNet.h"
#include <chrono>

using namespace std;

int InputCount = 1;
int NumLayers = 2;
int AFun[] = { TANH, LINE };
int NumNeurons[] = { 2, 1 };

Matrix2D TrainSet;

TrainParams Params;

double cur_fun(double x) {
	return x;
}

double cur_rand() {
	return rand() % 10 + 1;
}

Matrix2D generate_trainset(int NumInputs, int NumOutputs, int TrainsetSize) {
	Matrix2D Res(TrainsetSize, 2);
	for (int i = 0; i < TrainsetSize; ++i) {
		double inp = cur_rand();
		double out = cur_fun(double(i + 1));
		Res(i + 1, 1) = i + 1;
		Res(i + 1, 2) = out;
	}
	return Res;
}

Matrix2D generate_simset(int NumInputs, int SimSize) {
	Matrix2D Res(SimSize, NumInputs);
	for (int i = 0; i < SimSize; ++i) {
		double inp = cur_rand();
		Res(i + 1, 1) = i + 1;
	}
	return Res;
}

int main() {
	srand(time(nullptr));
	Net net(InputCount, NumLayers, AFun, NumNeurons);
	
	net.TrainSet(generate_trainset(1, 1, 100));
	Params.Error = 0.001;
	Params.MinGrad = 0.00000001;
	Params.NumEpochs = 20000;
	Params.Rate = 1e-3;
	vector<double> error;
	
	auto start_time = chrono::steady_clock::now();
	auto e = net.RMSPropTrain(Params, error);
	auto end_time = chrono::steady_clock::now();
	auto elapsed_ns = chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	cout << "Nanoseconds per epoch: " << elapsed_ns.count() / e << " ns" << endl;

	Matrix2D SimSet = generate_simset(1, 10);
	for (int i = 0; i < 10; ++i) {
		Matrix2D CurIn(1, 1), CurOut(1, 1);
		CurIn(1, 1) = SimSet(i + 1, 1);
		net.Simulate(CurIn, CurOut);
		cout << "In: " << CurIn(1, 1) << "\t\t Out: " << CurOut(1, 1) << endl;
	}
	system("pause");
	return 0;
}