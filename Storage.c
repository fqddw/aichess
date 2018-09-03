#include "NNLab.h"
#include "Storage.h"
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
int saveNeuralNetwork(NeuralNetWork* pNN,char* filename)
{
	FILE* fp = fopen(filename, "wb");
	fwrite(&pNN->inputWidth, sizeof(int), 1, fp);
	fwrite(&pNN->hiddenWidth, sizeof(int), 1, fp);
	fwrite(&pNN->hiddenCount, sizeof(int), 1, fp);
	fwrite(&pNN->outWidth, sizeof(int), 1, fp);
	fwrite(&pNN->dataLen, sizeof(int), 1, fp);
	fwrite(pNN->data, pNN->dataLen, 1, fp);
	fflush(fp);
	fclose(fp);
	return TRUE;
}

NeuralNetWork* loadNeuralNetwork(char* filename)
{
	FILE* fp = fopen(filename, "rb");
	if(!fp)
		return NULL;
	int inputWidth = 0;
	int hiddenWidth = 0;
	int hiddenCount = 0;
	int outWidth = 0;
	int dataLen = 0;
	int nRead = 0;
	nRead = fread(&inputWidth, sizeof(int), 1, fp);
	nRead = fread(&hiddenWidth, sizeof(int), 1, fp);
	nRead = fread(&hiddenCount, sizeof(int), 1, fp);
	nRead = fread(&outWidth, sizeof(int), 1, fp);
	nRead = fread(&dataLen, sizeof(int), 1, fp);
	double* data = (double*)malloc(dataLen);
	nRead = fread(data, dataLen, 1, fp);
	fclose(fp);
	return initNeuralNetworkManual(inputWidth, hiddenWidth, hiddenCount, outWidth, dataLen, data);
}
