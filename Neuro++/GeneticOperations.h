#pragma once

#include "MatrixOperations.h"
#include <vector>
#include <string>
#include <bitset>

using namespace std;

#define BYTE_ 8
#define RANDLIM 10000

#define TOURNAMENT 0
#define ROULETTE   1

// Стратегии формирования новой популяции
#define GCHP 0
#define GCH  1
#define LCHP 2
#define LCH  3

const double Prec = 1e-3; // Точность расчета весов и смещений

// Хромосома - цепочка закодированных признаков (весов и смещений)
////////////////////////////////////////////////////////////////////////////////
class Chromosome {
  protected:
    int MinVal; // Минимальное значение признаков (весов и смещений)
    int MaxVal; // Максимальное значение признаков (весов и смещений)
    double Precision; // Точность представления признаков (весов и смещений)
    unsigned long GrayCoding(unsigned int g) { return g ^ (g >> 1); }; // Кодирование кодом Грея
    unsigned long GrayDecoding(unsigned long g); // Декодирование кода Грея
  public:
    double FitnessValue; // Значение приспособленности для данной хромосомы
    double SelectionProbability; // Вероятность выбора в качестве родителя
    int GenotypeNum; // Число признаков (весов и смещений) в хромосоме
    int GeneNum; // Число генов в хромосоме
    vector<int> ChromosomePresentation; // Набор генов

    Chromosome(void); // Конструктор
    Chromosome(Matrix2D Phenotype, double Fitness); // Конструктор с параметрами
    void InitChromosome(Matrix2D Phenotype, double Fitness); // Инициализация хромосомы

    void Mutation(double MutationProbability); // Оператор мутации
    void Inversion(double InversionProbability); // Оператор инверсии
    Matrix2D GetPhenotype(); // Получение фенотипа хромосомы

    ~Chromosome() { }; // Деструктор
};
////////////////////////////////////////////////////////////////////////////////




// Популяция - набор хромосом
////////////////////////////////////////////////////////////////////////////////
class Population {
  protected:
    void Selection(int ParentsStrategy); // Выбор особей в качестве родителей
  public:
    int CalculateProbability(); // Расчет вероятностей выбора в качестве родителя для каждой особи
    int ChromosomesNum; // Число особей в популяции
    double AverageFitness; // Средняя приспособленность особей популяции
    std::vector<Chromosome> Chromosomes; // Набор хромосом
    Chromosome Father, Mother, Child1, Child2; // Хромосомы отца, матери и двух потомков

    Population(void) : ChromosomesNum(0) { }; // Конструктор
    void AddChromosome(Chromosome c); // Добавление особи в популяцию
    void SortPopulation(int Start, int End); // Сортировка особей популяции по возрастанию приспособленности
    void ClearPopulation(); // Очистка популяции
    void Crossingover(double CrossoverProbability, int ParentsStrategy); // Оператор кроссовера
    Chromosome SelectChild(int Strategy); // Выбор наиболее приспособленного потомка
};
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
string IntToBin(unsigned long n);
unsigned long BinToInt(string Bin);
////////////////////////////////////////////////////////////////////////////////

