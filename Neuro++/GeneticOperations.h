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

// ��������� ������������ ����� ���������
#define GCHP 0
#define GCH  1
#define LCHP 2
#define LCH  3

const double Prec = 1e-3; // �������� ������� ����� � ��������

// ��������� - ������� �������������� ��������� (����� � ��������)
////////////////////////////////////////////////////////////////////////////////
class Chromosome {
  protected:
    int MinVal; // ����������� �������� ��������� (����� � ��������)
    int MaxVal; // ������������ �������� ��������� (����� � ��������)
    double Precision; // �������� ������������� ��������� (����� � ��������)
    unsigned long GrayCoding(unsigned int g) { return g ^ (g >> 1); }; // ����������� ����� ����
    unsigned long GrayDecoding(unsigned long g); // ������������� ���� ����
  public:
    double FitnessValue; // �������� ����������������� ��� ������ ���������
    double SelectionProbability; // ����������� ������ � �������� ��������
    int GenotypeNum; // ����� ��������� (����� � ��������) � ���������
    int GeneNum; // ����� ����� � ���������
    vector<int> ChromosomePresentation; // ����� �����

    Chromosome(void); // �����������
    Chromosome(Matrix2D Phenotype, double Fitness); // ����������� � �����������
    void InitChromosome(Matrix2D Phenotype, double Fitness); // ������������� ���������

    void Mutation(double MutationProbability); // �������� �������
    void Inversion(double InversionProbability); // �������� ��������
    Matrix2D GetPhenotype(); // ��������� �������� ���������

    ~Chromosome() { }; // ����������
};
////////////////////////////////////////////////////////////////////////////////




// ��������� - ����� ��������
////////////////////////////////////////////////////////////////////////////////
class Population {
  protected:
    void Selection(int ParentsStrategy); // ����� ������ � �������� ���������
  public:
    int CalculateProbability(); // ������ ������������ ������ � �������� �������� ��� ������ �����
    int ChromosomesNum; // ����� ������ � ���������
    double AverageFitness; // ������� ����������������� ������ ���������
    std::vector<Chromosome> Chromosomes; // ����� ��������
    Chromosome Father, Mother, Child1, Child2; // ��������� ����, ������ � ���� ��������

    Population(void) : ChromosomesNum(0) { }; // �����������
    void AddChromosome(Chromosome c); // ���������� ����� � ���������
    void SortPopulation(int Start, int End); // ���������� ������ ��������� �� ����������� �����������������
    void ClearPopulation(); // ������� ���������
    void Crossingover(double CrossoverProbability, int ParentsStrategy); // �������� ����������
    Chromosome SelectChild(int Strategy); // ����� �������� ���������������� �������
};
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
string IntToBin(unsigned long n);
unsigned long BinToInt(string Bin);
////////////////////////////////////////////////////////////////////////////////

