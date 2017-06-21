#include "GeneticOperations.h"

using namespace std;

///////////////////////////////Хромомсома///////////////////////////////////////

// Конструктор
////////////////////////////////////////////////////////////////////////////////
Chromosome::Chromosome(void) {
	GenotypeNum = 0; GeneNum = 0;
	Precision = Prec;
	FitnessValue = 0; SelectionProbability = 0;
	MaxVal = static_cast<int>(floor(Precision*LONG_MAX / 2.0));
	MinVal = -MaxVal;
}
////////////////////////////////////////////////////////////////////////////////


// Конструктор с параметрами
////////////////////////////////////////////////////////////////////////////////
Chromosome::Chromosome(Matrix2D Phenotype, double Fitness) {
	int GeneCount = 0;
	GenotypeNum = Phenotype.GetRowCount();
	GeneNum = GenotypeNum * 4 * BYTE_;
	Precision = Prec;
	FitnessValue = Fitness; SelectionProbability = 0;
	MaxVal = static_cast<int>(floor(Precision*LONG_MAX / 2.0));
	MinVal = -MaxVal;
	ChromosomePresentation.resize(GeneNum);
	// Кодирование признаков
	// Каждый признак кодируется двоичной строкой в 32 бита
	for (int i = 1; i <= GenotypeNum; i++) {
		unsigned long Genotype = static_cast<unsigned long>(floor((Phenotype(i, 1) - MinVal) / Precision + 1));
		string GrayCode = IntToBin(GrayCoding(Genotype));
		for (size_t j = 1; j <= GrayCode.length(); j++)
			ChromosomePresentation[GeneCount++] = GrayCode[j] - '0';
	}
}
////////////////////////////////////////////////////////////////////////////////


// Инициализация хромосомы
////////////////////////////////////////////////////////////////////////////////
void Chromosome::InitChromosome(Matrix2D Phenotype, double Fitness) {
	int GeneCount = 0;
	GenotypeNum = Phenotype.GetRowCount();
	GeneNum = GenotypeNum * 4 * BYTE_;
	FitnessValue = Fitness; SelectionProbability = 0;
	Precision = Prec;
	MaxVal = static_cast<int>(floor(Precision*LONG_MAX / 2));
	MinVal = -MaxVal;
	ChromosomePresentation.resize(GeneNum);
	// Кодирование признаков
	// Каждый признак кодируется двоичной строкой в 32 бита
	for (int i = 1; i <= GenotypeNum; i++) {
		unsigned long Genotype = static_cast<unsigned long>(floor((Phenotype(i, 1) - MinVal) / Precision + 1));
		string GrayCode = IntToBin(GrayCoding(Genotype));
		for (size_t j = 1; j <= GrayCode.length(); j++)
			ChromosomePresentation[GeneCount++] = GrayCode[j] - '0';
	}
}
////////////////////////////////////////////////////////////////////////////////


// Получение фенотипа хромосомы
////////////////////////////////////////////////////////////////////////////////
Matrix2D Chromosome::GetPhenotype() {
	Matrix2D Phenotype(GenotypeNum, 1);
	string GrayCode = "";
	for (int i = 1; i <= GenotypeNum; i++) {
		GrayCode = "";
		for (int j = 4 * BYTE_*(i - 1); j < 4 * BYTE_*i; j++)
			GrayCode += to_string(ChromosomePresentation[j]);
		unsigned long n = GrayDecoding(BinToInt(GrayCode));
		Phenotype(i, 1) = MinVal + Precision*n - Precision / (double)2;
	}
	return Phenotype;
}
////////////////////////////////////////////////////////////////////////////////


// Декодирование кода Грея
////////////////////////////////////////////////////////////////////////////////
unsigned long Chromosome::GrayDecoding(unsigned long g) {
	unsigned long bin;
	for (bin = 0; g; g >>= 1) bin ^= g;
	return bin;
}
////////////////////////////////////////////////////////////////////////////////


// Оператор мутации
////////////////////////////////////////////////////////////////////////////////
void Chromosome::Mutation(double MutationProbability) {
	int IsMutation = 0, RndGene = 0;
	IsMutation = rand() % (RANDLIM + 1); // Вероятность мутации
	if (IsMutation > MutationProbability*RANDLIM) return;
	// Инвертирование случайно выбранного гена хромосомы
	RndGene = rand() % (GeneNum);
	ChromosomePresentation[RndGene] = !ChromosomePresentation[RndGene];
}
////////////////////////////////////////////////////////////////////////////////


// Оператор инверсии
////////////////////////////////////////////////////////////////////////////////
void Chromosome::Inversion(double InversionProbability) {
	int Pos = 0;
	double IsInversion = 0;
	IsInversion = rand() % (RANDLIM + 1); // Вероятность инверсии
	if (IsInversion > InversionProbability*RANDLIM) return;
	Pos = rand() % (GeneNum); // Точка инверсии
							  // Инверсия
	vector<int> Tmp(Pos + 1);
	for (int i = 0; i <= Pos; i++) Tmp[i] = ChromosomePresentation[i];
	for (int i = Pos + 1; i < GeneNum; i++)
		ChromosomePresentation[i - Pos - 1] = ChromosomePresentation[i];
	for (int i = GeneNum - Pos - 1; i < GeneNum; i++)
		ChromosomePresentation[i] = Tmp[i - GeneNum + Pos + 1];
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////




//////////////////////////////Класс популяции///////////////////////////////////

// Сортировка популяции по возрастанию приспособленности методом Хоара
////////////////////////////////////////////////////////////////////////////////
void Population::SortPopulation(int Start, int End) {
	int i = Start, j = End;
	double p = Chromosomes[End >> 1].FitnessValue;
	Chromosome tmp;
	do {
		while (Chromosomes[i].FitnessValue < p) i++;
		while (Chromosomes[j].FitnessValue > p) j--;
		if (i <= j) {
			tmp = Chromosomes[i];
			Chromosomes[i] = Chromosomes[j];
			Chromosomes[j] = tmp;
			i++; j--;
		}
	} while (i <= j);
	if (j > 0) SortPopulation(0, j);
	if (End > i) SortPopulation(i, End - i);
}
////////////////////////////////////////////////////////////////////////////////


// Очистка популяции
////////////////////////////////////////////////////////////////////////////////
void Population::ClearPopulation() {
	ChromosomesNum = 0;
	Chromosomes.resize(0);
}
////////////////////////////////////////////////////////////////////////////////


// Расчет вероятностей выбора в качестве родителя для каждой особи
////////////////////////////////////////////////////////////////////////////////
int Population::CalculateProbability() {
	double Sum = 0.0;
	int EndIndex = ChromosomesNum;
	for (int i = 0; i < ChromosomesNum; i++)
		AverageFitness += Chromosomes[i].FitnessValue;
	AverageFitness /= ChromosomesNum;
	for (int i = 0; i < ChromosomesNum; i++) {
		if (Chromosomes[i].FitnessValue > AverageFitness) { EndIndex = i; break; }
		Sum += 1.0 / Chromosomes[i].FitnessValue;
	}
	Chromosomes[0].SelectionProbability = 1.0 / Chromosomes[0].FitnessValue / Sum*RANDLIM;
	for (int i = 1; i < EndIndex; i++)
		Chromosomes[i].SelectionProbability =
		Chromosomes[i - 1].SelectionProbability + 1.0 / Chromosomes[i].FitnessValue / Sum*RANDLIM;
	return EndIndex;
}
////////////////////////////////////////////////////////////////////////////////

// Добавление особи в популяцию
////////////////////////////////////////////////////////////////////////////////
void Population::AddChromosome(Chromosome c) {
	Chromosomes.push_back(c); ChromosomesNum++;
}
////////////////////////////////////////////////////////////////////////////////


// Выбор особей в качестве родителей методом турнирного отбора
////////////////////////////////////////////////////////////////////////////////
void Population::Selection(int ParentsStrategy) {
	int RandNum = 0, EndIndex = 0;
	int i1 = 0, i2 = 0;
	double Fit1 = 0.0, Fit2 = 0.0;
	switch (ParentsStrategy) {
	case ROULETTE:   EndIndex = CalculateProbability();
		RandNum = rand() % (10001);
		if (RandNum <= Chromosomes[0].SelectionProbability)
			Father = Chromosomes[0];
		for (int i = 1; i < EndIndex; i++)
			if (RandNum <= Chromosomes[i].SelectionProbability &&
				RandNum > Chromosomes[i - 1].SelectionProbability)
			{
				Father = Chromosomes[i]; break;
			}
		RandNum = rand() % (10001);
		if (RandNum <= Chromosomes[0].SelectionProbability)
			Mother = Chromosomes[0];
		for (int i = 1; i < EndIndex; i++)
			if (RandNum <= Chromosomes[i].SelectionProbability &&
				RandNum > Chromosomes[i - 1].SelectionProbability)
			{
				Mother = Chromosomes[i]; break;
			}
		break;
	case TOURNAMENT: i1 = rand() % (ChromosomesNum);
		Fit1 = Chromosomes[i1].FitnessValue;
		i2 = rand() % (ChromosomesNum);
		Fit2 = Chromosomes[i2].FitnessValue;
		if (Fit1 < Fit2) Father = Chromosomes[i1];
		else Father = Chromosomes[i2];
		i1 = rand() % (ChromosomesNum);
		Fit1 = Chromosomes[i1].FitnessValue;
		i2 = rand() % (ChromosomesNum);
		Fit2 = Chromosomes[i2].FitnessValue;
		if (Fit1 < Fit2) Mother = Chromosomes[i1];
		else Mother = Chromosomes[i2];
		break;
	}
}
////////////////////////////////////////////////////////////////////////////////


// Оператор кроссовера
////////////////////////////////////////////////////////////////////////////////
void Population::Crossingover(double CrossoverProbability, int ParentsStrategy) {
	int CrossPosition = 0;
	double IsCrossover = 0.0;
	Selection(ParentsStrategy);
	Child1 = Mother; Child2 = Father;
	IsCrossover = (double)(rand() % (RANDLIM + 1));
	if (IsCrossover > CrossoverProbability*RANDLIM) return;
	CrossPosition = (rand() % (Mother.GeneNum - 1)) + 1;
	for (int j = CrossPosition; j < Mother.GeneNum; j++) {
		Child1.ChromosomePresentation[j] = Father.ChromosomePresentation[j];
		Child2.ChromosomePresentation[j] = Mother.ChromosomePresentation[j];
	}
}
////////////////////////////////////////////////////////////////////////////////


// Выбор наиболее приспособленного потомка
////////////////////////////////////////////////////////////////////////////////
Chromosome Population::SelectChild(int Strategy) {
	double MinFitness = Child1.FitnessValue;
	Chromosome Child = Child1;
	if (MinFitness > Child2.FitnessValue) { Child = Child2; MinFitness = Child2.FitnessValue; }
	if (Strategy == 3) return Child;
	if (MinFitness > Father.FitnessValue) { Child = Father; MinFitness = Father.FitnessValue; }
	if (MinFitness > Mother.FitnessValue) { Child = Child1; MinFitness = Child1.FitnessValue; }
	return Child;
}
////////////////////////////////////////////////////////////////////////////////


// Перевод целого числа в двоичный код
////////////////////////////////////////////////////////////////////////////////
string IntToBin(unsigned long n) {
	bitset<sizeof(unsigned long)> b(n);
	return b.to_string();
}
////////////////////////////////////////////////////////////////////////////////


//  Перевод двоичной строки в целое число
////////////////////////////////////////////////////////////////////////////////
unsigned long BinToInt(string Bin) {
	bitset<sizeof(unsigned long)> b(Bin);
	return b.to_ulong();
}
////////////////////////////////////////////////////////////////////////////////
