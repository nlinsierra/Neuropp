#include "NNet.h"

using namespace std;

bool StopTraining = false;

random_device rd;
mt19937 eng(rd());
uniform_int_distribution<> dist(1, 50000);


/////////////////////////// Функции класса слоев ///////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void Layer::Biases(const Matrix2D &b) {
	Biases__ = b;
	for (int i = 0; i < b.GetRowCount(); i++) Neurons(i).Bias(b(i + 1, 1));
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Layer::CalculateAxons(void) {
	for (auto &CurNeuron : Neurons()) CurNeuron.CalculateAxon();
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Layer::CalculateStates(Matrix2D Inp) {
	for (int i = 1; i <= Inp.GetRowCount(); i++) Neurons(i - 1).State(Inp(i, 1));
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление значений состояний нейронов слоя
//
// Взвешенные входы каждого нейрона суммируются, затем добавляется смещение, и
// полученный результат пропускается через активационную функцию
////////////////////////////////////////////////////////////////////////////////
void Layer::CalculateStates(const vector<Neuron> &PrevLayer) {
	double sum = 0.0;
	int i, j;
	for (i = 1; i <= NumNeurons(); i++) {
		for (j = 1; j <= PrevLayer.size(); j++)
			sum += Weights(i, j)*PrevLayer[j - 1].Axon();
		sum += Neurons(i - 1).Bias();
		Neurons(i - 1).State(sum);
		sum = 0.0;
	}
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////




/////////////////////////// Функции класса сетей ///////////////////////////////

////////////////////////////////////////////////////////////////////////////////
Net::Net(int NumInputs, int NumLayers, int LayerAFuns[], int LayerSizes[]) {
	InitInputs(NumInputs);
	InitLayers(NumInputs, NumLayers, LayerAFuns, LayerSizes);
	RandomWB();
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Net::InitInputs(int NumInputs) {
	InputsRanges__ = Matrix2D(NumInputs, 2);
	for (int i = 1; i <= NumInputs; i++) { 
		InputsRanges__(i, 1) = -1; 
		InputsRanges__(i, 2) = 1; 
	}
}
////////////////////////////////////////////////////////////////////////////////



// Инициализация слоев сети
////////////////////////////////////////////////////////////////////////////////
void Net::InitLayers(int NumInputs, int NumLayers, int AFun[], int NumLN[]) {
	Layers__.resize(NumLayers + 1);
	Matrix2D b, w, TmpW, TmpB, Range;
	double Imin = 0, Imax = 0;
	int CurLayerAFun, CurLayerSize, PrevLayerSize;
	// Нулевой слой - входы сети
	Layers__[0] = Layer(NumInputs, LINE);
	// Последующие слои
	for (int i = 1; i <= NumLayers; i++) {
		Layers__[i] = Layer(NumLN[i - 1], AFun[i - 1]);
		CurLayerAFun = Layers__[i].ActivateFunction();
		CurLayerSize = Layers__[i].NumNeurons();
		PrevLayerSize = Layers__[i - 1].NumNeurons();
		Imin = RangeIn[0][CurLayerAFun];
		Imax = RangeIn[1][CurLayerAFun];
		if (i == 1) Range = InputsRanges__;
		else {
			Range = Matrix2D(PrevLayerSize, 2);
			for (int j = 1; j <= PrevLayerSize; j++) {
				Range(j, 1) = RangeOut[0][Layers__[i - 1].ActivateFunction()];
				Range(j, 2) = RangeOut[1][Layers__[i - 1].ActivateFunction()];
			}
		}
		w = Matrix2D(CurLayerSize, PrevLayerSize);
		b = Matrix2D(CurLayerSize, 1);
		// Инициализация весов и смещений сети методом Нгуена-Видроу
		for (int j = 1; j <= CurLayerSize; j++) {
			for (int k = 1; k <= PrevLayerSize; k++)
				w(j, k) = SignedRandomVal(1.0);
			b(j, 1) = SignedRandomVal(1.0);
		}
		if (CurLayerAFun != LINE) {
			double Beta = 0.7*pow(CurLayerSize, 1.0 / (double)PrevLayerSize);
			Matrix2D NormaW(CurLayerSize, 1);
			for (int j = 1; j <= CurLayerSize; j++) {
				for (int k = 1; k <= PrevLayerSize; k++)
					NormaW(j, 1) += w(j, k)*w(j, k);
				NormaW(j, 1) = sqrt(NormaW(j, 1));
				if (NormaW(j, 1) == 0.0) NormaW(j, 1) = 1;
			}
			for (int j = 1; j <= CurLayerSize; j++)
				for (int k = 1; k <= PrevLayerSize; k++)
					w(j, k) *= Beta / NormaW(j, 1);
			for (int j = 1; j <= b.GetRowCount(); j++) b(j, 1) = SignedRandomVal(Beta);
			double x = 0.5*(Imax - Imin), y = 0.5*(Imax + Imin);
			w = w*x; b = b*x + y;
			Matrix2D a(PrevLayerSize, 1), c(PrevLayerSize, 1);
			for (int j = 1; j <= PrevLayerSize; j++) {
				a(j, 1) = 2.0 / (Range(j, 2) - Range(j, 1));
				c(j, 1) = 1.0 - Range(j, 2)*a(j, 1);
			}
			b = w*c + b;
			for (int j = 1; j <= CurLayerSize; j++) for (int k = 1; k <= PrevLayerSize; k++)
				w(j, k) = w(j, k)*a(k, 1);
		}
		Layers__[i].Biases(b);
		Layers__[i].Weights(w);
	}
	// Заполнение единого вектора весов и смещений
	WeightsBiases__ = Matrix2D(GetNumWeights() + GetNumBiases(), 1);
	TmpW = VectorToMatrix2D(GetWeights(), GetNumWeights(), 1);
	TmpB = VectorToMatrix2D(GetBiases(), GetNumBiases(), 1);
	for (int i = 1; i <= GetNumWeights(); i++) WeightsBiases__(i, 1) = TmpW(i, 1);
	for (int i = GetNumWeights() + 1; i <= GetNumWeights() + GetNumBiases(); i++)
		WeightsBiases__(i, 1) = TmpB(i - GetNumWeights(), 1);
	TrainSet__ = Matrix2D(1, NumInputs + NumOutputs());
}
////////////////////////////////////////////////////////////////////////////////




// Перевод единого вектора весов и смещений в веса и смещения слоев и нейронов
////////////////////////////////////////////////////////////////////////////////
void Net::WBNetToLayers() {
	Matrix2D Wt(GetNumWeights(), 1), Bs(GetNumBiases(), 1), w, b;
	int WCount = 1, BCount = 1, count = 1;
	// Разбиение единого вектора на вектора весов и смещений
	for (int i = 1; i <= GetNumWeights(); i++)
		Wt(i, 1) = WeightsBiases(i, 1);
	for (int i = GetNumWeights() + 1; i <= WeightsBiases__.GetRowCount(); i++)
		Bs(i - GetNumWeights(), 1) = WeightsBiases(i, 1);
	// Идентификация весов и смещений слоев
	for (int i = 1; i <= NumLayers(); i++) {
		w = Matrix2D(Layers__[i].NumNeurons()*Layers__[i - 1].NumNeurons(), 1);
		count = 1;
		for (int j = WCount; j < WCount + w.GetRowCount(); j++)
			w(count++, 1) = Wt(j, 1);
		WCount += w.GetRowCount();
		Layers__[i].Weights(VectorToMatrix2D(w.Matrix2DToVector(), Layers__[i].NumNeurons(), Layers__[i - 1].NumNeurons()));
		b = Matrix2D(Layers__[i].NumNeurons(), 1);
		count = 1;
		for (int j = BCount; j < BCount + b.GetRowCount(); j++)
			b(count++, 1) = Bs(j, 1);
		BCount += b.GetRowCount();
		Layers__[i].Biases(b);
	}
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление размерности вектора весовых коэффициентов
////////////////////////////////////////////////////////////////////////////////
int Net::GetNumWeights() {
	int count = 0;
	for (int i = 1; i <= NumLayers(); i++)
		count += Layers__[i - 1].NumNeurons()*Layers__[i].NumNeurons();
	return count;
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление размерности вектора смещений
////////////////////////////////////////////////////////////////////////////////
int Net::GetNumBiases() {
	int count = 0;
	for (int i = 1; i <= NumLayers(); i++) count += Layers__[i].NumNeurons();
	return count;
}
////////////////////////////////////////////////////////////////////////////////


// Возврат вектора весовых коэффициентов
////////////////////////////////////////////////////////////////////////////////
vector<double> Net::GetWeights() {
	vector<double> w;
	for (int i = 1; i <= NumLayers(); i++)
		for (int j = 1; j <= Layers__[i].NumNeurons(); j++)
			for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++)
				w.push_back(Layers__[i].Weights(j, k));
	return w;
}
////////////////////////////////////////////////////////////////////////////////


// Возврат вектора смещений нейронов
////////////////////////////////////////////////////////////////////////////////
vector<double> Net::GetBiases() {
	vector<double> b;
	for (int i = 1; i <= NumLayers(); i++)
		for (int j = 0; j < Layers__[i].NumNeurons(); j++)
			b.push_back(Layers__[i].Neurons(j).Bias());
	return b;
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление градиента вектора ошибки по смещениям
////////////////////////////////////////////////////////////////////////////////
vector<Matrix2D> Net::CalculateDelta(Matrix2D SimOut, Matrix2D AimOut) {
	vector<Matrix2D> Delta(NumLayers());
	// Дельта для выходного слоя нейронов
	double Diff = 0.0;
	Delta[NumLayers() - 1] = move(Matrix2D(NumOutputs(), 1));
	for (int i = 1; i <= NumOutputs(); i++) {
		Diff = Layers__[NumLayers()].Neurons(i - 1).Derivative();
		Delta[NumLayers() - 1](i, 1) = (SimOut(i, 1) - AimOut(i, 1))*Diff;
	}
	// Расчет для остальных слоев сети
	for (int LayerCount = NumLayers() - 1; LayerCount > 0; LayerCount--) {
		Delta[LayerCount - 1] = Matrix2D(Layers__[LayerCount].NumNeurons(), 1);
		for (int i = 1; i <= Delta[LayerCount - 1].GetRowCount(); i++) {
			// Суммирование дельта нейронов следующего слоя
			double Sum = 0;
			for (int NCount = 1; NCount <= Layers__[LayerCount + 1].NumNeurons(); NCount++)
				Sum += Delta[LayerCount](NCount, 1)*Layers__[LayerCount + 1].Weights(NCount, i);
			// Вычисление дельта нейронов текущего слоя
			Diff = Layers__[LayerCount].Neurons(i - 1).Derivative();
			Delta[LayerCount - 1](i, 1) = Sum*Diff;
		}
	}
	return move(Delta);
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление градиента
////////////////////////////////////////////////////////////////////////////////
Matrix2D Net::CalculateGradient(Matrix2D SimOut, Matrix2D AimOut) {
	vector<Matrix2D> Delta, Gradient(NumLayers());
	Matrix2D ResultGrad(GetNumWeights() + GetNumBiases(), 1);
	double Axon = 0.0;
	int count = 1, NumNeurons = 0, NumPrevNeurons = 0;
	Delta = CalculateDelta(SimOut, AimOut);
	NumNeurons = NumOutputs();
	NumPrevNeurons = Layers__[NumLayers() - 1].NumNeurons();
	Gradient[NumLayers() - 1] = Matrix2D(NumNeurons, NumPrevNeurons);
	for (int i = 1; i <= NumNeurons; i++)
		for (int j = 1; j <= NumPrevNeurons; j++) {
			Axon = Layers__[NumLayers() - 1].Neurons(j - 1).Axon();
			Gradient[NumLayers() - 1](i, j) = Delta[NumLayers() - 1](i, 1)*Axon;
		}
	for (int LayerCount = NumLayers() - 1; LayerCount > 0; LayerCount--) {
		NumNeurons = Layers__[LayerCount].NumNeurons();
		NumPrevNeurons = Layers__[LayerCount - 1].NumNeurons();
		Gradient[LayerCount - 1] = Matrix2D(NumNeurons, NumPrevNeurons);
		for (int i = 1; i <= NumNeurons; i++)
			for (int j = 1; j <= NumPrevNeurons; j++) {
				Axon = Layers__[LayerCount - 1].Neurons(j - 1).Axon();
				Gradient[LayerCount - 1](i, j) = Delta[LayerCount - 1](i, 1)*Axon;
			}
	}
	for (int n = 1; n <= NumLayers(); n++)
		for (int i = 1; i <= Layers__[n].NumNeurons(); i++)
			for (int j = 1; j <= Layers__[n - 1].NumNeurons(); j++)
				ResultGrad(count++, 1) = Gradient[n - 1](i, j);
	for (int n = 1; n <= NumLayers(); n++)
		for (int i = 1; i <= Layers__[n].NumNeurons(); i++)
			ResultGrad(count++, 1) = Delta[n - 1](i, 1);
	return move(ResultGrad);
}
////////////////////////////////////////////////////////////////////////////////


// Нахождение суммарного градиента
////////////////////////////////////////////////////////////////////////////////
Matrix2D Net::SumGradient() {
	int PairCount = 1;
	Matrix2D Out(NumOutputs(), 1), Inp(NumInputs(), 1);
	Matrix2D ResOut(NumOutputs(), 1), Grad, TmpGrad;
	while (PairCount <= TrainSet__.GetRowCount()) {
		// Выделение очередной обучающей пары
		for (int i = 1; i <= NumInputs(); i++) Inp(i, 1) = TrainSet(PairCount, i);
		for (int i = NumInputs() + 1; i <= NumInputs() + NumOutputs(); i++)
			Out(i - NumInputs(), 1) = TrainSet(PairCount, i);
		// Вычисление выходов сети
		Simulate(Inp, ResOut);
		TmpGrad = CalculateGradient(ResOut, Out);
		if (PairCount == 1) Grad = move(TmpGrad);
		else Grad = Grad + TmpGrad;
		PairCount++;
	}
	return move(Grad);
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление значения функции ошибки по всем обучающим парам
////////////////////////////////////////////////////////////////////////////////
double Net::CalculateError() {
	int PairCount = 1;
	Matrix2D Out(NumOutputs(), 1), Inp(NumInputs(), 1);
	Matrix2D ResOut(NumOutputs(), 1);
	double CurrentError = 0.0;
	while (PairCount <= TrainSet__.GetRowCount()) {
		// Выделение очередной обучающей пары
		for (int i = 1; i <= NumInputs(); i++) Inp(i, 1) = TrainSet(PairCount, i);
		for (int i = NumInputs() + 1; i <= NumInputs() + NumOutputs(); i++)
			Out(i - NumInputs(), 1) = TrainSet(PairCount, i);
		// Вычисление выходов сети
		Simulate(Inp, ResOut);
		// Вычисление текущей ошибки по всем выходам
		for (int i = 1; i <= NumOutputs(); i++)
			CurrentError += pow(ResOut(i, 1) - Out(i, 1), 2) / 2.0;
		PairCount++;
	}
	return CurrentError;
}
////////////////////////////////////////////////////////////////////////////////


// Моделирование работы нейронной сети
//
// Последовательно вычисляются аксоны для каждого слоя,
// затем аксоны последнего слоя сохраняются в массив, расположенный
// по адресу, передаваемому через параметр Outputs
////////////////////////////////////////////////////////////////////////////////
void Net::Simulate(Matrix2D Inputs, Matrix2D &Outputs) {
	Layers__[0].CalculateStates(Inputs);
	Layers__[0].CalculateAxons();
	for (int i = 1; i <= NumLayers(); i++) {
		Layers__[i].CalculateStates(Layers__[i - 1].Neurons());
		Layers__[i].CalculateAxons();
	}
	for (int j = 1; j <= NumOutputs(); j++)
		Outputs(j, 1) = Layers__[NumLayers()].Neurons(j - 1).Axon();
}
////////////////////////////////////////////////////////////////////////////////


// Обучение сети он-лайн - градиентный алгоритм
// Коррекция весов и смещений при подаче каждой обучающей пары
////////////////////////////////////////////////////////////////////////////////
int Net::GradTrainOnLine(TrainParams Params, vector<double> &Error) {
	Matrix2D Inp(NumInputs(), 1), Out(NumOutputs(), 1);
	Matrix2D ResOut(NumOutputs(), 1), Grad;
	int PairCount = 1;
	double CurrentError = 0, Norma = 0;
	// Внешний цикл - по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		PairCount = 1; CurrentError = 0;
		// Внутренний цикл - по обучающим парам
		while (PairCount <= TrainSet__.GetRowCount()) {
			// Выделение очередной обучающей пары
			for (int i = 1; i <= NumInputs(); i++) Inp(i, 1) = TrainSet(PairCount, i);
			for (int i = NumInputs() + 1; i <= NumInputs() + NumOutputs(); i++)
				Out(i - NumInputs(), 1) = TrainSet(PairCount, i);
			// Вычисление выходов сети
			Simulate(Inp, ResOut);
			// Вычисление текущей ошибки по всем выходам
			for (int i = 1; i <= NumOutputs(); i++)
				CurrentError += pow(ResOut(i, 1) - Out(i, 1), 2);
			Grad = CalculateGradient(ResOut, Out);
			Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
			if (Norma <= Params.MinGrad) return l;
			Grad = Grad / Norma;
			// Коррекция весов и смещений
			WeightsBiases__ = WeightsBiases__ - Params.Rate*Grad;
			WBNetToLayers();
			PairCount++;
		}
		Error.push_back(CurrentError / 2);
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Обучение сети офф-лайн - градиентный алгоритм
// Коррекция весов и смещений после предъявления всей обучающей выборки
////////////////////////////////////////////////////////////////////////////////
int Net::GradTrainOffLine(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad;
	double Norma = 0.0;
	// Цикл по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		Grad = SumGradient();
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma <= Params.MinGrad) return l;
		Grad = Grad / Norma;
		WeightsBiases__ = WeightsBiases__ - Params.Rate*Grad;
		WBNetToLayers();
		Error.push_back(CalculateError());
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Алгоритм наискорейшего спуска он-лайн
////////////////////////////////////////////////////////////////////////////////
int Net::FastGradTrainOnLine(TrainParams Params, vector<double> &Error) {
	double CurrentError = 0.0, Norma = 0.0;
	Matrix2D Out(NumOutputs(), 1), Inp(NumInputs(), 1);
	Matrix2D ResOut(NumOutputs(), 1), Grad, PrevWB, Tmp;
	int PairCount = 1;
	PrevWB = WeightsBiases__;
	// Внешний цикл - по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		PairCount = 1; CurrentError = 0;
		// Внутренний цикл - по обучающим парам
		while (PairCount <= TrainSet__.GetRowCount()) {
			// Выделение очередной обучающей пары
			for (int i = 1; i <= NumInputs(); i++) Inp(i, 1) = TrainSet(PairCount, i);
			for (int i = NumInputs() + 1; i <= NumInputs() + NumOutputs(); i++)
				Out(i - NumInputs(), 1) = TrainSet(PairCount, i);
			// Вычисление выходов сети
			Simulate(Inp, ResOut);
			// Вычисление текущей ошибки по всем выходам
			for (int i = 1; i <= NumOutputs(); i++)
				CurrentError += pow(ResOut(i, 1) - Out(i, 1), 2);
			// Вычисление градиента для коррекции весовых коэффициентов
			Grad = CalculateGradient(ResOut, Out);
			Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
			if (Norma <= Params.MinGrad) return l;
			Grad = Grad / Norma;
			// Коррекция весов и смещений
			Tmp = WeightsBiases__ - PrevWB;
			PrevWB = WeightsBiases__;
			WeightsBiases__ = WeightsBiases__ - Params.Rate*Grad + Params.Alpha*Tmp;
			WBNetToLayers();
			PairCount++;
		}
		Error.push_back(CurrentError / 2.0);
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


//  Алгоритм наискорейшего спуска офф-лайн
////////////////////////////////////////////////////////////////////////////////
int Net::FastGradTrainOffLine(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad, PrevWB, Tmp;
	double Norma = 0.0;
	PrevWB = WeightsBiases__;
	// Цикл по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		Grad = SumGradient();
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma <= Params.MinGrad) return l;
		Grad = Grad / Norma;
		Tmp = WeightsBiases__ - PrevWB;
		PrevWB = WeightsBiases__;
		WeightsBiases__ = WeightsBiases__ - Params.Rate*Grad + Params.Alpha*Tmp;
		WBNetToLayers();
		Error.push_back(CalculateError());
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Нахождение параметра обучения посредством метода золотого сечения
////////////////////////////////////////////////////////////////////////////////
double Net::GoldSection(double &MinVal, double &MaxVal, Matrix2D Direction) {
	double phi = 0.618, Val = 1.0, Error1 = 0.0, Error2 = 0.0, Eps = 0.001;
	double Val1 = MaxVal - (MaxVal - MinVal)*phi, Val2 = MinVal + (MaxVal - MinVal)*phi;
	Matrix2D WBOld = WeightsBiases__;
	WeightsBiases__ = WeightsBiases__ + Val1*Direction;
	WBNetToLayers();
	Error1 = CalculateError();
	WeightsBiases__ = WBOld + Val2*Direction;
	WBNetToLayers();
	Error2 = CalculateError();
	while (MaxVal - MinVal > Eps) if (Error1 <= Error2) {
		MaxVal = Val2; Val2 = Val1; Error2 = Error1;
		Val1 = MaxVal - (MaxVal - MinVal)*phi;
		WeightsBiases__ = WBOld + Val1*Direction;
		WBNetToLayers();
		Error1 = CalculateError();
		Val = MinVal;
	}
	else {
		MinVal = Val1; Val1 = Val2; Error1 = Error2;
		Val2 = MinVal + (MaxVal - MinVal)*phi;
		WeightsBiases__ = WBOld + Val2*Direction;
		WBNetToLayers();
		Error2 = CalculateError();
		Val = MaxVal;
	}
	return Val;
}
////////////////////////////////////////////////////////////////////////////////


// Алгоритм сопряженных градиентов Флетчера-Ривса
////////////////////////////////////////////////////////////////////////////////
int Net::FRConjugateGradTrain(TrainParams Params, vector<double> &Error) {
	double Beta, Sigma0, SigmaNew, SigmaOld, Val1, Val2, Val, Norma, Eps = 0.001;
	Matrix2D Grad, Direction, WBOld;
	int VarCount, EpochCount = 0;
	Grad = -SumGradient();
	Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
	Grad = Grad / Norma;
	VarCount = Grad.GetRowCount();
	Direction = Grad;
	SigmaNew = (Grad.Transpose()*Grad)(1, 1); Sigma0 = SigmaNew;
	// Цикл по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		if (SigmaNew <= Eps*Eps*Sigma0) return l;
		WBOld = WeightsBiases__;
		// Одномерная минимизация скорости обучения
		Val1 = 0; Val2 = 1; Val = GoldSection(Val1, Val2, Direction);
		WeightsBiases__ = WBOld + Val*Direction; WBNetToLayers();
		// Вычисление нового градиента
		Grad = -SumGradient();
		// Нахождение нормы градиента
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		// Условие достижения минимума градиента
		if (Norma < Params.MinGrad) return l;
		Grad = Grad / Norma;
		// Нахождение нового направления поиска
		SigmaOld = SigmaNew; SigmaNew = (Grad.Transpose()*Grad)(1, 1);
		Beta = SigmaNew / SigmaOld;
		Direction = Grad + Beta*Direction;
		EpochCount++;
		if (EpochCount == VarCount || (Grad.Transpose()*Direction)(1, 1) <= 0) {
			Direction = Grad; EpochCount = 0;
		}
		Error.push_back(CalculateError());
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Алгоритм RPROP
////////////////////////////////////////////////////////////////////////////////
int Net::RPropTrain(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad, PrevGrad, Delta;
	double EttaPlus = 1.2, EttaMinus = 0.5, DeltaMax = 50.0, DeltaMin = 1e-6, Norma;
	Grad = SumGradient();
	Delta = move(Matrix2D(Grad.GetRowCount(), 1));
	for (int l = 0; l < Params.NumEpochs; l++) {
		PrevGrad = move(Grad);
		Grad = SumGradient();
#ifdef DEBUG_PRINT
		cout << "Grad before correction:" << endl;
		Grad.Transpose().ShowElements();
		cout << endl;
#endif
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma < Params.MinGrad) return l;
#ifdef DEBUG_PRINT
		WeightsBiases__.Transpose().ShowElements();
		cout << endl;
#endif
		for (int i = 1; i <= Grad.GetRowCount(); i++) {
			if (l == 0) Delta(i, 1) = RandomVal(1.0);
			else if (PrevGrad(i, 1)*Grad(i, 1) > 0.0) Delta(i, 1) = min(Delta(i, 1)*EttaPlus, DeltaMax);
			else if (PrevGrad(i, 1)*Grad(i, 1) < 0.0) Delta(i, 1) = max(Delta(i, 1)*EttaMinus, DeltaMin);
			WeightsBiases__(i, 1) += -Delta(i, 1)*sign(Grad(i, 1));
		}
#ifdef DEBUG_PRINT
		cout << "Delta:" << endl;
		Delta.Transpose().ShowElements();
		cout << endl;
#endif
		WBNetToLayers();
#ifdef DEBUG_PRINT
		cout << "After correction:" << endl;
		WeightsBiases__.Transpose().ShowElements();
#endif
		Error.push_back(CalculateError());
#ifdef DEBUG_PRINT
		cout << Error[l] << endl << endl;
#endif
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Алгоритм RMSPROP
////////////////////////////////////////////////////////////////////////////////
int Net::RMSPropTrain(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad, RMS, PrevRMS, Delta, G, PrevG;
	double Gamma = 0.95, Eps = 1e-2, Norma;
	Grad = SumGradient();
	RMS = Matrix2D(Grad.GetRowCount(), 1);
	G = Matrix2D(Grad.GetRowCount(), 1);
	Delta = Matrix2D(Grad.GetRowCount(), 1);
	for (int l = 0; l < Params.NumEpochs; l++) {
		PrevRMS = RMS;
		PrevG = G;
		Grad = SumGradient();
		G = Gamma*PrevG + (1 - Gamma)*Grad;
		RMS = Gamma*PrevRMS + (1 - Gamma)*Grad*Grad;
#ifdef DEBUG_PRINT
		cout << "Grad before correction:" << endl;
		Grad.Transpose().ShowElements();
		cout << endl;
#endif
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma < Params.MinGrad) return l;
#ifdef DEBUG_PRINT
		WeightsBiases__.Transpose().ShowElements();
		cout << endl;
#endif
		for (int i = 1; i <= Grad.GetRowCount(); i++) {
			Delta(i, 1) = Params.Rate * Grad(i, 1) / sqrt(RMS(i, 1) - G(i, 1)*G(i, 1) + Eps);
			WeightsBiases__(i, 1) += -Delta(i, 1);
		}
#ifdef DEBUG_PRINT
		cout << "Delta:" << endl;
		Delta.Transpose().ShowElements();
		cout << endl;
#endif
		WBNetToLayers();
#ifdef DEBUG_PRINT
		cout << "After correction:" << endl;
		WeightsBiases__.Transpose().ShowElements();
#endif
		Error.emplace_back(CalculateError());
//#ifdef DEBUG_PRINT
		if (l % 100) cout << l << ":\t\t" << Error[l] << endl << endl;
//#endif
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление ошибки обучения по выходу NumOut и обучающей паре PairCount
////////////////////////////////////////////////////////////////////////////////
double Net::CalculateOutputError(int NumOut, int PairCount) {
	double CurrentError = 0.0, Out;
	Matrix2D Inp(NumInputs(), 1), ResOut(NumOutputs(), 1);
	for (int j = 1; j <= NumInputs(); j++) Inp(j, 1) = TrainSet(PairCount, j);
	Out = TrainSet(PairCount, NumInputs() + NumOut);
	Simulate(Inp, ResOut);
	CurrentError = ResOut(NumOut, 1) - Out;
	return CurrentError;
}
////////////////////////////////////////////////////////////////////////////////


// Вычисление Якобиана функции ошибки
////////////////////////////////////////////////////////////////////////////////
Matrix2D Net::CalculateJacobian() {
	int NumPairs = TrainSet__.GetRowCount();
	int NumWB = WeightsBiases__.GetRowCount(), Count = 0;
	Matrix2D Jacobian(NumOutputs() + NumPairs, NumWB), Inp(NumInputs(), 1), ResOut;
	double CurrentError = 0.0, Step = 0.01;
	for (int i = 1; i <= NumPairs; i++) {
		for (int k = 1; k <= NumOutputs(); k++) {
			CurrentError = CalculateOutputError(k, i);
			Count++;
			for (int j = 1; j <= NumWB; j++) {
				WeightsBiases__(j, 1) += Step; WBNetToLayers();
				Jacobian(Count, j) = (CalculateOutputError(k, i) - CurrentError) / Step;
				WeightsBiases__(j, 1) -= Step;
			}
			WBNetToLayers();
		}
	}
	return Jacobian;
}
////////////////////////////////////////////////////////////////////////////////


// Алгоритм Левенберга-Марквардта
////////////////////////////////////////////////////////////////////////////////
int Net::LMTrain(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad, J, H, I, OldWB, DiagH;
	double Lambda = 0.005, Norma, PrevError;
	I = CalculateOnes(WeightsBiases__.GetRowCount());
	PrevError = CalculateError();
	for (int l = 0; l < Params.NumEpochs; l++) {
		J = CalculateJacobian(); H = J.Transpose()*J;
		Grad = SumGradient(); Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma < Params.MinGrad) return l;
		while (Lambda <= 1e10) {
			OldWB = WeightsBiases__;
			Matrix2D Tmp = H + Lambda*I;
			WeightsBiases__ = WeightsBiases__ - Tmp.Inverse()*Grad;
			WBNetToLayers();
			Error.push_back(CalculateError());
			if (Error.back() < PrevError) {
				PrevError = Error.back(); Lambda /= 10.0;
				if (Lambda < 1e-20) Lambda = 1e-20;
				break;
			}
			WeightsBiases__ = OldWB; WBNetToLayers();
			Lambda *= 10.0;
		}
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


// Байесовская регуляризация
////////////////////////////////////////////////////////////////////////////////
int Net::BayesianReg(TrainParams Params, vector<double> &Error) {
	Matrix2D Grad, I, J, H, OldWB;
	double Alpha, Beta, Norma, WBError, Lambda = 0.005, PrevError, Gamma, CurrentError;
	int NumPairs = TrainSet__.GetRowCount();
	int NumWB = WeightsBiases__.GetRowCount(), NumErrors = NumOutputs()*NumPairs;
	I = CalculateOnes(NumWB);
	// Инициализация параметров
	CurrentError = CalculateError();
	WBError = (WeightsBiases__.Transpose()*WeightsBiases__)(1, 1);
	Gamma = NumWB;
	Alpha = Gamma / (2 * WBError);
	Beta = (NumErrors - Gamma) / (2.0 * CurrentError);
	if (Beta <= 0) Beta = 1;
	// Инициализация регуляризационной функции ошибки
	PrevError = Beta*CurrentError + Alpha*WBError;
	// Цикл по эпохам
	for (int l = 0; l < Params.NumEpochs; l++) {
		// Вычисление якобиана и гессиана функции ошибки
		J = CalculateJacobian(); H = J.Transpose()*J;
		// Вычисление градиента функции ошибки
		Grad = SumGradient();
		Norma = sqrt((Grad.Transpose()*Grad)(1, 1));
		if (Norma < Params.MinGrad) return l;
		// Минимизация регуляризационной функции ошибки по
		// алгоритму Левенберга-Марквардта
		while (Lambda <= 1e10) {
			// Сохранение предыдущих значений весов/смещений сети
			OldWB = WeightsBiases__;
			// Коррекция весов/смещений
			Matrix2D Tmp = Beta*H + (Lambda + Alpha)*I;
			Matrix2D RegGrad = Beta*Grad + Alpha*WeightsBiases__;
			WeightsBiases__ = WeightsBiases__ - Tmp.Inverse()*RegGrad; WBNetToLayers();
			// Нахождение суммы квадратов весов/смещений
			WBError = (WeightsBiases__.Transpose()*WeightsBiases__)(1, 1);
			// Вычисление регуляризационной функции ошибки
			CurrentError = Beta*CalculateError() + Alpha*WBError;
			// Если ошибка уменьшается то завершаем итерацию
			if (CurrentError < PrevError) {
				Lambda /= 10.0;
				if (Lambda < 1e-10) Lambda = 1e-10;
				break;
			}
			// Если ошибка увеличивается, то возвращаем старые значения весов/смещений
			WeightsBiases__ = OldWB; WBNetToLayers();
			Lambda *= 10.0;
		}
		if (Lambda <= 1e10) {
			// Вычисление новых значений параметров регуляризации
			J = CalculateJacobian(); H = J.Transpose()*J;
			Gamma = NumWB - Alpha*(Beta*H + Alpha*I).Inverse().Trace();
			Alpha = Gamma / (2.0 * WBError);
			CurrentError = CalculateError();
			Beta = (NumErrors - Gamma) / (2.0 * CurrentError);
			// Вычисление регуляризационной функции ошибки
			PrevError = Beta*CurrentError + Alpha*WBError;
			Error.push_back(CurrentError);
		}
		else return l;
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Net::RandomWB() {
	Matrix2D b, w, TmpW, TmpB, Range;
	double Imin = 0.0, Imax = 0.0;
	for (int i = 1; i <= NumLayers(); i++) {
		Imin = RangeIn[0][Layers__[i].ActivateFunction()];
		Imax = RangeIn[1][Layers__[i].ActivateFunction()];
		if (i == 1) Range = InputsRanges__;
		else {
			Range = Matrix2D(Layers__[i - 1].NumNeurons(), 2);
			for (int j = 1; j <= Layers__[i - 1].NumNeurons(); j++) {
				Range(j, 1) = RangeOut[0][Layers__[i - 1].ActivateFunction()];
				Range(j, 2) = RangeOut[1][Layers__[i - 1].ActivateFunction()];
			}
		}
		w = Matrix2D(Layers__[i].NumNeurons(), Layers__[i - 1].NumNeurons());
		b = Matrix2D(Layers__[i].NumNeurons(), 1);

		// Инициализация весов и смещений сети методом Нгуена-Видроу
		if (Layers__[i].ActivateFunction() == LINE) {
			for (int j = 1; j <= Layers__[i].NumNeurons(); j++)
				for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++) w(j, k) = SignedRandomVal(1);
			for (int j = 1; j <= b.GetRowCount(); j++) b(j, 1) = SignedRandomVal(1);
		}
		else {
			for (int j = 1; j <= Layers__[i].NumNeurons(); j++)
				for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++) w(j, k) = SignedRandomVal(1);
			double Beta = 0.7*pow(Layers__[i].NumNeurons(), 1 / (double)Layers__[i - 1].NumNeurons());
			Matrix2D NormaW(Layers__[i].NumNeurons(), 1);
			for (int j = 1; j <= Layers__[i].NumNeurons(); j++) {
				for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++)
					NormaW(j, 1) += pow(w(j, k), 2);
				NormaW(j, 1) = sqrt(NormaW(j, 1));
				if (NormaW(j, 1) == 0) NormaW(j, 1) = 1;
			}
			for (int j = 1; j <= Layers__[i].NumNeurons(); j++)
				for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++)
					w(j, k) *= Beta / NormaW(j, 1);
			for (int j = 1; j <= b.GetRowCount(); j++) b(j, 1) = SignedRandomVal(Beta);
			double x = 0.5*(Imax - Imin), y = 0.5*(Imax + Imin);
			w = w*x; b = b*x + y;
			Matrix2D a(Layers__[i - 1].NumNeurons(), 1), c(Layers__[i - 1].NumNeurons(), 1);
			for (int j = 1; j <= Layers__[i - 1].NumNeurons(); j++) {
				a(j, 1) = 2 / (Range(j, 2) - Range(j, 1));
				c(j, 1) = 1 - Range(j, 2)*a(j, 1);
			}
			b = w*c + b;
			for (int j = 1; j <= Layers__[i].NumNeurons(); j++) for (int k = 1; k <= Layers__[i - 1].NumNeurons(); k++)
				w(j, k) = w(j, k)*a(k, 1);
		}
		Layers__[i].Biases(b);
		Layers__[i].Weights(w);
	}
	TmpW = VectorToMatrix2D(GetWeights(), GetNumWeights(), 1);
	TmpB = VectorToMatrix2D(GetBiases(), GetNumBiases(), 1);
	for (int i = 1; i <= GetNumWeights(); i++)
		WeightsBiases__(i, 1) = TmpW(i, 1);
	for (int i = GetNumWeights() + 1; i <= GetNumWeights() + GetNumBiases(); i++)
		WeightsBiases__(i, 1) = TmpB(i - GetNumWeights(), 1);
#ifdef DEBUG_PRINT
	WeightsBiases__.Transpose().ShowElements();
	cout << endl;
#endif
}
////////////////////////////////////////////////////////////////////////////////




// Генетический алгоритм
////////////////////////////////////////////////////////////////////////////////
int Net::GeneticTraining(TrainParams Params, vector<double> &Error) {
	Population OldPopulation, NewPopulation;
	int PopulationSize = Params.PopSize;
	double Fitness = 0.0;
	Chromosome NewChromosome;
	for (int i = 0; i < PopulationSize; i++) {
		RandomWB();
		Fitness = CalculateError();
		NewChromosome.InitChromosome(WeightsBiases__, Fitness);
		OldPopulation.AddChromosome(NewChromosome);
	}
	OldPopulation.SortPopulation(0, PopulationSize - 1);
	for (int l = 0; l < Params.NumEpochs; l++) {
		for (int i = 0; i < PopulationSize; i++) {
			// Кроссовер
			OldPopulation.Crossingover(Params.CrossoverP, Params.ParentsStrategy);
			// Оценка приспособленности потомков
			WeightsBiases__ = OldPopulation.Child1.GetPhenotype(); WBNetToLayers();
			OldPopulation.Child1.FitnessValue = CalculateError();
			WeightsBiases__ = OldPopulation.Child2.GetPhenotype(); WBNetToLayers();
			OldPopulation.Child2.FitnessValue = CalculateError();
			// Выбор стратегии формирования новой популяции
			switch (Params.SelStrategy) {
				// Глобальный отбор с участием родителей
			case GCHP: // Инверсия
				OldPopulation.Child1.Inversion(Params.InversionP);
				OldPopulation.Child2.Inversion(Params.InversionP);
				// Мутация
				OldPopulation.Child1.Mutation(Params.MutationP);
				OldPopulation.Child2.Mutation(Params.MutationP);
				// Добавление потомка в новую популяцию
				NewPopulation.AddChromosome(OldPopulation.Child1);
				NewPopulation.AddChromosome(OldPopulation.Child2);
				NewPopulation.AddChromosome(OldPopulation.Mother);
				NewPopulation.AddChromosome(OldPopulation.Father);
				break;
				// Глобальный отбор без участия родителей
			case GCH:  // Инверсия
				OldPopulation.Child1.Inversion(Params.InversionP);
				OldPopulation.Child2.Inversion(Params.InversionP);
				// Мутация
				OldPopulation.Child1.Mutation(Params.MutationP);
				OldPopulation.Child2.Mutation(Params.MutationP);
				// Добавление потомка в новую популяцию
				NewPopulation.AddChromosome(OldPopulation.Child1);
				NewPopulation.AddChromosome(OldPopulation.Child2);
				break;
				// Локальный отбор
			case LCHP:
			case LCH:  // Отбор потомка с наименьшим фитнесом
				NewChromosome = OldPopulation.SelectChild(Params.SelStrategy);
				// Инверсия
				NewChromosome.Inversion(Params.InversionP);
				// Мутация
				NewChromosome.Mutation(Params.MutationP);
				// Добавление потомка в новую популяцию
				NewPopulation.AddChromosome(NewChromosome);
				break;
			}

		}
		NewPopulation.SortPopulation(0, PopulationSize - 1);
		OldPopulation.ClearPopulation();
		for (int i = 0; i < PopulationSize; i++)
			OldPopulation.AddChromosome(NewPopulation.Chromosomes[i]);
		Error.push_back(OldPopulation.Chromosomes[0].FitnessValue);
		WeightsBiases__ = OldPopulation.Chromosomes[0].GetPhenotype();
		WBNetToLayers();
		NewPopulation.ClearPopulation();
		if (Error.back() <= Params.Error) return l;
		if (StopTraining) return l;
	}
	return Params.NumEpochs - 1;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void Elman::InitLayers(int NumInputs, int NumLayers, int AFun[], int NumLN[]) {
	if (NumLayers >= 2) ContextNeuronCount = NumLN[NumLayers - 2];
	else ContextNeuronCount = 0;
	Net::InitLayers(NumInputs + ContextNeuronCount, NumLayers, AFun, NumLN);
}


Matrix2D Elman::Simulate(Matrix2D Inputs, Matrix2D &Outputs) {
	Net::Simulate(Inputs, Outputs);
	return GetHiddenAxons();
}

Matrix2D Elman::Simulate(Matrix2D Inputs, Matrix2D &Outputs, Matrix2D ContextInputs) {
	Matrix2D NewInputs(Inputs.GetRowCount() + ContextInputs.GetRowCount(), Inputs.GetColCount());
	for (int i = 1; i <= Inputs.GetRowCount(); ++i)
		for (int j = 1; j <= Inputs.GetColCount(); ++j)
			NewInputs(i, j) = Inputs(i, j);

	for (int i = 1; i <= ContextInputs.GetRowCount(); ++i)
		for (int j = 1; j <= ContextInputs.GetColCount(); ++j)
			NewInputs(i + Inputs.GetRowCount(), j) = ContextInputs(i, j);

	Net::Simulate(NewInputs, Outputs);
	return GetHiddenAxons();
}



// Дополнительные математические функции
////////////////////////////////////////////////////////////////////////////////
int sign(double x) {
	if (x < 0) return -1;
	else if (x > 0) return 1;
	else return 0;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
double SignedRandomVal(double Val) {
	double res = 0;
	res = double(dist(eng) % 1001) / double(1000) - double(dist(eng) % (1001)) / double(1000);
	res *= Val;
	return res;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
double RandomVal(double Val) {
	double res = 0;
	res = double(dist(eng) % (1001)) / double(1000);
	res *= Val;
	return res;
}
////////////////////////////////////////////////////////////////////////////////
