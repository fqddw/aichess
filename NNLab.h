#ifndef __NNLAB_H__
#define __NNLAB_H__
#define LNSTEP 0.5

typedef struct _palparam PALPARAM;
typedef struct Matrix_
{
	int rowCount;
	int colCount;
	double* data;
}Matrix;
typedef struct _Vector
{
	int size;
	double* data;
}Vector;
typedef struct _doubleArr
{
	double delta;
	double value;
}doubleArr;
typedef struct _VectorTrain
{
	int size;
	doubleArr* data;
}VectorTrain;

typedef struct _VectorTrainArray
{
	int size;
	VectorTrain* arr;
}VectorTrainArray;

typedef struct NeuralNetWork_ {
	int inputWidth;
	int hiddenWidth;
	int hiddenCount;
	int outWidth;
	int dataLen;
	double* data;
	Matrix** net;
	Vector** bias;
}NeuralNetWork;

typedef struct Record_
{
	Vector* pInput;
	Vector* pOutput;
}Record;

typedef struct RecordSet_
{
	int size;
	Record* pRecord;
}RecordSet;


NeuralNetWork* initNeuralNetwork(int, int, int, int);
NeuralNetWork* initNeuralNetworkManual(int, int, int, int, int, double*);
VectorTrainArray* initVectorTrainArray(int, int, int);
VectorTrainArray* forward_train(PALPARAM*, NeuralNetWork*, Vector*, VectorTrainArray*);
void backward(PALPARAM*, NeuralNetWork*, VectorTrainArray*, Vector*, Vector*);

VectorTrainArray* every_forward_train(NeuralNetWork*, Vector*, VectorTrainArray*, int, int);
void every_backward(NeuralNetWork*, VectorTrainArray*, Vector*, Vector*, int, int);
Vector* initVector(int);
Vector* initVectorManual(int,double*);

#include "ThreadPool.h"
#endif
