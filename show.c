#include "stdio.h"
#include "NNLab.h"
#include "Storage.h"


int main(int argc, char** argv)
{
	char* filename = argv[1];
	NeuralNetWork* pNN = loadNeuralNetwork(filename);
	printf("i%d hl%d hc%d o%d\n", pNN->inputWidth, pNN->hiddenCount, pNN->hiddenWidth, pNN->outWidth);
	int j = 0;
	for(;j< pNN->hiddenWidth; j++)
	{
		int i = 0;
		for(;i<pNN->inputWidth;i++)
		{
			printf("%d %d %.32lf\n", i, j, pNN->data[i + j*pNN->inputWidth]);
		}
		printf("\n");
	}
}
