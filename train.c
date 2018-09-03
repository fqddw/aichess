#include "NNLab.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "unistd.h"
#include "Storage.h"
#include "math.h"
#include "NNsl.h"
#include "chess.h"

Record* loadData(char* filename)
{
	int offset = 0;
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	int count = 10000;
	double* pData = (double*)malloc(count*92*sizeof(double));
	int nsize = fread(pData, count*92*sizeof(double), 1, fp);
	int i = 0;
	Record* pRecord= (Record*)malloc(count*sizeof(Record));
	for(;i<count;i++)
	{
		(pRecord+i)->pInput = initVectorManual(91, pData+offset);
		offset+=91;
		(pRecord+i)->pOutput = initVectorManual(1, pData+offset);
		offset+=1;
	}
	fclose(fp);
	return pRecord;
}
int getfilesize(char* filename)
{
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fclose(fp);
	return len;
}
double* loadRawData(char* filename)
{
	int offset = 0;
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	int count = 10000;
	double* pData = (double*)malloc(len);
	//double* pData = (double*)malloc(count*92*sizeof(double));
	int nsize = fread(pData, len, 1, fp);
	//int nsize = fread(pData, count*92*sizeof(double), 1, fp);
	fclose(fp);
	return pData;
}

int main(int argc,char** argv)
{
	srand(12124141);
	NeuralNetWork* pNN = loadNeuralNetwork("nn.txt");
	if(!pNN)
		pNN = initNeuralNetwork(91, 100, 2, 1);
	//Record* pRecord = loadData("cr.qp") ;
	PALPARAM* params = (PALPARAM*)malloc(sizeof(PALPARAM));
	params->pNN = pNN;
	params->layer = 0;
	pthread_mutex_init(&params->m, NULL);
	sem_init(&params->evm, 0, 0);
	sem_init(&params->evs, 0, 0);
	//create_thread_pool(params);
	int count = 800;
	int i = 0;
	VectorTrainArray* pArray = initVectorTrainArray(pNN->hiddenCount, pNN->hiddenWidth, pNN->outWidth);
	params->pArray = pArray;
	int it = 0;
	int ct = 0;
	double* pData = loadRawData("cr.qp");
	/*pData = (double*)malloc(1000*sizeof(double)*3);
	for(;i<3*1000;i+=3)
	{
		pData[i] = (double)rand()/RAND_MAX;
		pData[i+1] = (double)rand()/RAND_MAX;
		if(pData[i+1]>pow(pData[i],3))
			pData[i+2] = 1.0;
		else
			pData[i+2]=0.0;

		printf("%lf %lf %lf\n", pData[i], pData[i+1], pData[i+2]);
	}*/
	int totalcount = getfilesize("cr.qp")/(sizeof(double)*92);
	int traincount = totalcount*0.95;
	double* result = (double*)malloc(2*sizeof(double)*(pNN->hiddenWidth*pNN->hiddenCount+pNN->hiddenWidth*pNN->outWidth));
	while(1)
	{
	train_g(pNN->data, pData, result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth, traincount);
	saveNeuralNetwork(pNN, "nn.txt");
	int testi = 0;
	double error = 0.0;
	for(;testi<traincount;testi++)
	{
		forward_g(pNN->data, pData+testi*(pNN->inputWidth+pNN->outWidth), result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);
error+=pow(result[pNN->hiddenWidth*pNN->hiddenCount*2]-pData[(pNN->inputWidth+pNN->outWidth)*testi+pNN->inputWidth],2);
	}

	printf("train error %.16lf\n", error/traincount);
	testi = traincount;
	double trainerror = error;
	error = 0.0;
	for(;testi<totalcount;testi++)
	{
		forward_g(pNN->data, pData+testi*(pNN->inputWidth+pNN->outWidth), result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);
		error+=pow(result[pNN->hiddenWidth*pNN->hiddenCount*2]-pData[(pNN->inputWidth+pNN->outWidth)*testi+pNN->inputWidth],2);
	}
	printf("test error %.16lf\n", error/(totalcount-traincount));
	if(trainerror/traincount<0.06)
	{
		v_main();
		free(pData);
		pData = loadRawData("cr.qp");
		totalcount = getfilesize("cr.qp")/(sizeof(double)*92);
		traincount = totalcount*0.95;
	}

	ct++;
	}
	/*double* result = (double*)malloc(2*sizeof(double)*(pNN->hiddenWidth*pNN->hiddenCount+pNN->hiddenWidth*pNN->outWidth));
	while(1)
	{
	train_g(pNN->data, pData, result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth, 10000);
	saveNeuralNetwork(pNN, "nn.txt");
	int testi = 0;
	double error = 0.0;
	for(;testi<9500;testi++)
	{
		forward_g(pNN->data, pData+testi*(pNN->inputWidth+pNN->outWidth), result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);
error+=pow(result[pNN->hiddenWidth*pNN->hiddenCount*2]-pData[92*testi+91],2);
	}

	testi = 9500;
	for(;testi<10000;testi++)
	{
		forward_g(pNN->data, pData+testi*(pNN->inputWidth+pNN->outWidth), result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);
		printf("test val %.16lf %lf\n", result[pNN->hiddenWidth*pNN->hiddenCount*2], pData[92*testi+91]);
	}
	printf("error %.16lf\n", error/9500);
	ct++;
	}*/
	/*while(1)
	{
		i=0;
	while(i<801)
	{
		params->pRecord = pRecord+i;
		forward_train(params, pNN, params->pRecord->pInput, pArray);
		//printf("train value %.16lf %.16lf\n",pArray->arr[2].data->value, params->pRecord->pOutput->data[0]);
		//params->type = BACKWARD;
		//backward(params, pNN, pArray, params->pRecord->pInput, params->pRecord->pOutput);
		i++;
	}
	i=800;
	double error = 0.0;
	while(i<1000)
	{
		params->pRecord = pRecord+i;
		params->type = FORWARD;
		forward_train(params, pNN, params->pRecord->pInput, pArray);
		if(i == 900 || i == 953)
		printf("test  value %d %d %d %d %.16lf %.16lf %d\n",pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth, pArray->arr[pNN->hiddenCount].data->value, params->pRecord->pOutput->data[0], i);
		error+=pow(params->pRecord->pOutput->data[0]-pArray->arr[pNN->hiddenCount].data->value, 2);
		i++;
	}
	printf("%.16lf\n", error/200);
	if(it%1000 == 0)
	{
		//saveNeuralNetwork(pNN, "nn.txt");
		it = 1;
	}
	else
		it++;
	}*/
}
