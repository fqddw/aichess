#include "NNLab.h"
#include "NNsl.h"
#include "Storage.h"
#include "math.h"
#include "malloc.h"

#include "stdio.h"
double* loadRawData(char* filename)
{
	int offset = 0;
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	int count = 10000;
	double* pData = (double*)malloc(count*92*sizeof(double));
	int nsize = fread(pData, count*92*sizeof(double), 1, fp);
	return pData;
}

int main()
{
	double* pData = loadRawData("crtest.qp");
	NeuralNetWork* pNN = loadNeuralNetwork("nn.txt");
	double* result = (double*)malloc(2*sizeof(double)*(pNN->hiddenWidth*pNN->hiddenCount+pNN->hiddenWidth*pNN->outWidth));
	int i=0;
	double error = 0.0;
	while(i<10000)
	{
		forward_g(pNN->data, pData+i*(pNN->inputWidth+pNN->outWidth), result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);
		printf("%lf %lf\n", result[pNN->hiddenWidth*pNN->hiddenCount*2], pData[(pNN->inputWidth+pNN->outWidth)*i+pNN->inputWidth]);
		error+=pow(result[pNN->hiddenWidth*pNN->hiddenCount*2]-pData[(pNN->inputWidth+pNN->outWidth)*i+pNN->inputWidth],2);
		i++;
	}
	printf("error %lf\n", error/10000);
}
