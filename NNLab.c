#include "stdlib.h"
#include "time.h"
#include "math.h"
#include "memory.h"
#include "stdio.h"
#include "unistd.h"
#include "NNLab.h"
double sigmoid(double val)
{
	return 1.0 / (1.0 + exp(-val));
}
Matrix* initMatrix(int rowCount, int colCount)
{
	Matrix* pMatrix = (Matrix*)malloc(sizeof(Matrix));
	pMatrix->rowCount = rowCount;
	pMatrix->colCount = colCount;
	pMatrix->data = (double*)malloc(rowCount*colCount * sizeof(double));
	for (int i = 0; i < rowCount*colCount;i++)
	{
		int flag = rand() % 2;
		if (flag == 0)
			flag = -1;
		*(pMatrix->data+i) = 0;//(rand())*flag;
	}
	/*
	
	memset(pMatrix->data, 1.0/rand(), rowCount*colCount * sizeof(double));

	*/
	return pMatrix;
}
Matrix* initMatrixManual(int rowCount, int colCount, double* data)
{
	Matrix* pMatrix = (Matrix*)malloc(sizeof(Matrix));
	pMatrix->rowCount = rowCount;
	pMatrix->colCount = colCount;
	pMatrix->data = data;
	return pMatrix;
}

Vector* initVector(int size)
{
	Vector* pVector = (Vector*)malloc(sizeof(Vector));
	pVector->size = size;
	pVector->data = (double*)malloc(size * sizeof(double));
	memset(pVector->data, 0, size * sizeof(double));
	return pVector;
}
Vector* initVectorManual(int size, double* data)
{
	Vector* pVector = (Vector*)malloc(sizeof(Vector));
	pVector->size = size;
	pVector->data = data;
	return pVector;
}

NeuralNetWork* initNeuralNetwork(int inputWidth, int hiddenWidth, int hiddenCount, int outWidth)
{
	NeuralNetWork* pNeuralNetWork = (NeuralNetWork*)malloc(sizeof(NeuralNetWork));
	pNeuralNetWork->inputWidth = inputWidth;
	pNeuralNetWork->hiddenWidth = hiddenWidth;
	pNeuralNetWork->hiddenCount = hiddenCount;
	pNeuralNetWork->outWidth = outWidth;
	int itemCount = (inputWidth*hiddenWidth)+(hiddenCount-1)*(hiddenWidth*hiddenWidth)+(hiddenWidth*outWidth)+(hiddenCount-1)*hiddenWidth+outWidth;
	int dataLen = sizeof(double)*((inputWidth*hiddenWidth)+(hiddenCount-1)*(hiddenWidth*hiddenWidth)+(hiddenWidth*outWidth)+(hiddenCount-1)*hiddenWidth+outWidth);
	pNeuralNetWork->dataLen = dataLen;
	pNeuralNetWork->data = (double*)malloc(dataLen);
	if(!pNeuralNetWork->data)
		exit(0);
	memset(pNeuralNetWork->data,0,pNeuralNetWork->dataLen);
	int rd = 0;
	for(;rd<itemCount;rd++)
	{
		pNeuralNetWork->data[rd] = 1.0/(rand()+1);
	}
	pNeuralNetWork->net = (Matrix**)malloc((hiddenCount + 1) * sizeof(Matrix*));
	/*pNeuralNetWork->net[0] = initMatrix(inputWidth, hiddenWidth);
	pNeuralNetWork->net[hiddenCount] = initMatrix(hiddenWidth, outWidth);
	for (int i = 1; i < hiddenCount; i++)
	{
		pNeuralNetWork->net[i] = initMatrix(hiddenWidth, hiddenWidth);
	}

	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVector(hiddenWidth);
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVector(outWidth);
	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVector(hiddenWidth);
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVector(outWidth);
	return pNeuralNetWork;*/

	int offset = 0;
	pNeuralNetWork->net[0] = initMatrixManual(inputWidth, hiddenWidth, pNeuralNetWork->data + offset);
	offset += inputWidth*hiddenWidth;
	for (int i = 1; i < hiddenCount; i++)
	{
		pNeuralNetWork->net[i] = initMatrixManual(hiddenWidth, hiddenWidth, pNeuralNetWork->data+offset);
		offset += hiddenWidth*hiddenWidth;
	}
	pNeuralNetWork->net[hiddenCount] = initMatrixManual(hiddenWidth, outWidth, pNeuralNetWork->data+offset);
	offset += hiddenWidth*outWidth;

	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVectorManual(hiddenWidth, pNeuralNetWork->data+offset);
		offset += hiddenWidth;
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVectorManual(outWidth, pNeuralNetWork->data+offset);

	return pNeuralNetWork;
}

NeuralNetWork* initNeuralNetworkManual(int inputWidth, int hiddenWidth, int hiddenCount, int outWidth, int dataLen, double* data)
{
	NeuralNetWork* pNeuralNetWork = (NeuralNetWork*)malloc(sizeof(NeuralNetWork));
	pNeuralNetWork->inputWidth = inputWidth;
	pNeuralNetWork->hiddenWidth = hiddenWidth;
	pNeuralNetWork->hiddenCount = hiddenCount;
	pNeuralNetWork->outWidth = outWidth;
	pNeuralNetWork->dataLen = dataLen;
	pNeuralNetWork->data = data;
	if(!pNeuralNetWork->data)
		exit(0);
	pNeuralNetWork->net = (Matrix**)malloc((hiddenCount + 1) * sizeof(Matrix*));
	/*pNeuralNetWork->net[0] = initMatrix(inputWidth, hiddenWidth);
	pNeuralNetWork->net[hiddenCount] = initMatrix(hiddenWidth, outWidth);
	for (int i = 1; i < hiddenCount; i++)
	{
		pNeuralNetWork->net[i] = initMatrix(hiddenWidth, hiddenWidth);
	}

	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVector(hiddenWidth);
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVector(outWidth);
	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVector(hiddenWidth);
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVector(outWidth);*/

	int offset = 0;
	pNeuralNetWork->net[0] = initMatrixManual(inputWidth, hiddenWidth, pNeuralNetWork->data + offset);
	offset += inputWidth*hiddenWidth;
	for (int i = 1; i < hiddenCount; i++)
	{
		pNeuralNetWork->net[i] = initMatrixManual(hiddenWidth, hiddenWidth, pNeuralNetWork->data+offset);
		offset += hiddenWidth*hiddenWidth;
	}
	pNeuralNetWork->net[hiddenCount] = initMatrixManual(hiddenWidth, outWidth, pNeuralNetWork->data+offset);
	offset += hiddenWidth*outWidth;

	pNeuralNetWork->bias = (Vector**)malloc(sizeof(Vector*)*pNeuralNetWork->hiddenCount);
	for(int i=0;i<pNeuralNetWork->hiddenCount-1;i++)
	{
		*(pNeuralNetWork->bias+i) = initVectorManual(hiddenWidth, pNeuralNetWork->data+offset);
		offset += hiddenWidth;
	}
	*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1) = initVectorManual(outWidth, pNeuralNetWork->data+offset);

	return pNeuralNetWork;
}



void releaseVector(Vector* pVector)
{
	free(pVector->data);
	free(pVector);
}

VectorTrainArray* initVectorTrainArray(int count, int width, int outputsize)
{
	VectorTrainArray* pArray = (VectorTrainArray*)malloc(sizeof(VectorTrainArray));
	pArray->size = count+1;
	pArray->arr = (VectorTrain*)malloc((1+count)*sizeof(VectorTrain));
	int i = 0;
	for(i=0;i<count;i++)
	{
		(pArray->arr+i)->size = width;
		(pArray->arr+i)->data = (doubleArr*)malloc(width*sizeof(doubleArr));
	}
	(pArray->arr+count)->size = outputsize;
	(pArray->arr+count)->data = (doubleArr*)malloc(outputsize*sizeof(doubleArr));
	return pArray;
}


VectorTrainArray* forward_train(PALPARAM* params,NeuralNetWork* pNeuralNetWork, Vector* pVectorInput, VectorTrainArray* pArray)
{
	/*params->type = FORWARD;
	params->layer = 0;
	params->layerundone = params->pNN->hiddenCount+1;

	while(1)
	{
		if(params->layerundone == 0)
		{
			return 0;
		}
		else
		{
			params->nindex = 0;
			if(params->layer == pNeuralNetWork->hiddenCount)
			{
				params->nindexundone = pNeuralNetWork->outWidth;
				params->max_nindex = pNeuralNetWork->outWidth;
			}
			else
			{
				params->nindexundone = pNeuralNetWork->hiddenWidth;
				params->max_nindex = pNeuralNetWork->hiddenWidth;
			}
			sem_post(&params->evs);
		}
		sem_wait(&params->evm);
		params->layerundone -= 1;
		params->layer += 1;
	}*/
	int i = 0;
	int j = 0;
	int k = 0;
	VectorTrain* hiddenVector = pArray->arr;

	int index = 0;
	
	for (j = 0; j < pNeuralNetWork->hiddenWidth; j++) {
		(hiddenVector->data+j)->value = 0;
		for (i = 0; i<pVectorInput->size; i++)
		{
			(hiddenVector->data+j)->value += (*(pVectorInput->data+i)) * (*((*pNeuralNetWork->net)->data+pVectorInput->size*j + i));
		}
		(hiddenVector->data+j)->value = sigmoid((hiddenVector->data+j)->value);
		(hiddenVector->data+j)->delta = 0.0;
	}
	index++;

	for (i = 1; i<pNeuralNetWork->hiddenCount; i++)
	{
		for (j = 0; j < pNeuralNetWork->hiddenWidth; j++) {
			((hiddenVector+index)->data+j)->value = 0.0;
			for (k = 0; k < pNeuralNetWork->hiddenWidth; k++) {
				((hiddenVector+index)->data+j)->value += ((hiddenVector+index-1)->data+k)->value * (*(*(pNeuralNetWork->net+i))->data+pNeuralNetWork->hiddenWidth*j + k);
			}
			((hiddenVector+index)->data+j)->value += *((*(pNeuralNetWork->bias+i-1))->data+j);
			((hiddenVector+index)->data+j)->value = sigmoid(((hiddenVector+index)->data+j)->value);
		}
		index++;
	}
	index = pNeuralNetWork->hiddenCount;
	
	for (j = 0; j < pNeuralNetWork->outWidth; j++) {
		((hiddenVector+index)->data+j)->value = 0.0;
		for (i = 0; i<pNeuralNetWork->hiddenWidth; i++)
		{
			((hiddenVector+index)->data+j)->value += ((hiddenVector+index-1)->data+i)->value * (*(*(pNeuralNetWork->net+pNeuralNetWork->hiddenCount))->data+pNeuralNetWork->hiddenWidth*j + i);
		}
		((hiddenVector+index)->data+j)->value += *((*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1))->data+j);
		((hiddenVector+index)->data+j)->value = sigmoid(((hiddenVector+index)->data+j)->value);
	}
	//releaseVector(hiddenVector);
	return pArray;
}

VectorTrainArray* every_forward_train(NeuralNetWork* pNeuralNetWork, Vector* pVectorInput, VectorTrainArray* pArray, int layer, int nindex)
{
	int i = 0;
	int j = 0;
	int k = 0;
	VectorTrain* hiddenVector = pArray->arr;

	int index = layer;
	
	j = nindex;
	if(layer == 0){
		(hiddenVector->data+j)->value = 0;
		for (i = 0; i<pVectorInput->size; i++)
		{
			(hiddenVector->data+j)->value += (*(pVectorInput->data+i)) * (*((*pNeuralNetWork->net)->data+pVectorInput->size*j + i));
		}
		(hiddenVector->data+j)->value = sigmoid((hiddenVector->data+j)->value);
		(hiddenVector->data+j)->delta = 0.0;
	}else if(layer!=pNeuralNetWork->hiddenCount){

			((hiddenVector+index)->data+j)->value = 0.0;
			for (k = 0; k < pNeuralNetWork->hiddenWidth; k++) {
				((hiddenVector+index)->data+j)->value += ((hiddenVector+index-1)->data+k)->value * (*(*(pNeuralNetWork->net+i))->data+pNeuralNetWork->hiddenWidth*j + k);
			}
			((hiddenVector+index)->data+j)->value += *((*(pNeuralNetWork->bias+index-1))->data+j);
			((hiddenVector+index)->data+j)->value = sigmoid(((hiddenVector+index)->data+j)->value);
	}else{
		((hiddenVector+index)->data+j)->value = 0.0;
		for (i = 0; i<pNeuralNetWork->hiddenWidth; i++)
		{
			((hiddenVector+index)->data+j)->value += (hiddenVector->data+i)->value * (*(*(pNeuralNetWork->net+pNeuralNetWork->hiddenCount))->data+pNeuralNetWork->hiddenWidth*j + i);
		}
		((hiddenVector+index)->data+j)->value += *((*(pNeuralNetWork->bias+pNeuralNetWork->hiddenCount-1))->data+j);
		((hiddenVector+index)->data+j)->value = sigmoid(((hiddenVector+index)->data+j)->value);
	}
	//releaseVector(hiddenVector);
	return pArray;
}


Vector* getOutputDeltaVector(Vector* pOutput, Vector* pTarget)
{
	Vector* pDelta = initVector(pOutput->size);
	int i = 0;
	for(;i<pOutput->size;i++)
	{
		double Ok = pOutput->data[i];
		double yk = pTarget->data[i];
		pDelta->data[i] = (1.0-Ok)*Ok*(Ok-yk);
	}
	return pDelta;
}

void updateTrainCol(VectorTrainArray* pArray, int index)
{
	if(index == pArray->size - 1)
	{
	}
	(pArray->arr+index)->data->delta =(pArray->arr+index+1)->data->delta = 1;
}

/*
VectorTrainArray* forward_train(NeuralNetWork* pNeuralNetWork, Vector* pVectorInput)
{
	VectorTrainArray* pArray = initVectorTrainArray(pNeuralNetWork->hiddenCount, pNeuralNetWork->hiddenWidth);
	int i = 0;
	int j = 0;
	int k = 0;
	VectorTrain* hiddenVector = pArray->arr;
	VectorTrain* hiddenVectorRes = pArray->arr;
	VectorTrain* hiddenVectorTmp = hiddenVector;
	VectorTrain* outVector = NULL;
	
	for (j = 0; j < pNeuralNetWork->hiddenWidth; j++) {
		for (i = 0; i<pVectorInput->size; i++)
		{
			hiddenVector->data[j] += pVectorInput->data[i] * pNeuralNetWork->net[0]->data[pVectorInput->size*j + i];
		}
		hiddenVector->data[j] = sigmoid(hiddenVector->data[j]);
	}

	for (i = 1; i<pNeuralNetWork->hiddenCount; i++)
	{
		for (j = 0; j < pNeuralNetWork->hiddenWidth; j++) {
			for (k = 0; k < pNeuralNetWork->hiddenWidth; k++) {
				hiddenVectorRes->data[j] += hiddenVector->data[k] * pNeuralNetWork->net[i]->data[hiddenVector->size*j + k];
			}
			hiddenVectorRes->data[j] += pNeuralNetWork->bias->data[i];
			hiddenVectorRes->data[j] = sigmoid(hiddenVectorRes->data[j]);
		}
		hiddenVectorTmp = hiddenVector;
		hiddenVector = hiddenVectorRes;
		hiddenVectorRes = hiddenVectorTmp;
	}
	
	for (j = 0; j < pNeuralNetWork->outWidth; j++) {
		for (i = 0; i<pNeuralNetWork->hiddenWidth; i++)
		{
			outVector->data[j] += hiddenVector->data[i] * pNeuralNetWork->net[pNeuralNetWork->hiddenCount]->data[hiddenVector->size*j + i];
		}
		outVector->data[j] = sigmoid(outVector->data[j]);
	}
	releaseVector(hiddenVectorRes);
	releaseVector(hiddenVector);
	return pArray;
}
*/

void calvariance(RecordSet* pRecordSet, Vector* pOutput)
{
}
void backward(PALPARAM* params, NeuralNetWork* pNeuralNetWork, VectorTrainArray* pArray, Vector* pInput, Vector* pTarget)
{
	/*params->type = BACKWARD;
	params->layer = pNeuralNetWork->hiddenCount;
	params->layerundone = params->pNN->hiddenCount+1;
	while(1)
	{
		pthread_mutex_lock(&params->m);
		if(params->layerundone == 0)
		{
			pthread_mutex_unlock(&params->m);
			return;
		}
		else
		{
			params->nindex = 0;
			if(params->layer == pNeuralNetWork->hiddenCount)
			{
				params->nindexundone = pNeuralNetWork->outWidth;
				params->max_nindex = pNeuralNetWork->outWidth;
			}
			if(params->layer == -1)
			{
				params->nindexundone = pNeuralNetWork->inputWidth;
				params->max_nindex = pNeuralNetWork->inputWidth;
			}
			else
			{
				params->nindexundone = pNeuralNetWork->hiddenWidth+1;
				params->max_nindex = pNeuralNetWork->hiddenWidth+1;
			}
			pthread_mutex_unlock(&params->m);
			sem_post(&params->evs);
		}
		sem_wait(&params->evm);
		params->layerundone -= 1;
		params->layer -= 1;

	}*/

	VectorTrain* pOutputVectorTrain = pArray->arr+pNeuralNetWork->hiddenCount;
	int it = 0;
	for(;it<pNeuralNetWork->outWidth;it++)
	{
		double Ok = (pOutputVectorTrain->data+it)->value;
		double Tk = *(pTarget->data+it);
		(pOutputVectorTrain->data+it)->delta=(1.0-Ok)*Ok*(Ok-Tk);
	}
	int i = pNeuralNetWork->hiddenCount-1;
	for(;i>=0;i--)
	{
		VectorTrain* pVectorTrain= pArray->arr+i;
		int index = 0;
		for(;index<pVectorTrain->size;index++)
		{
			double delta = 0.0;
			int nexti = 0;
			for(;nexti<(pArray->arr+i+1)->size;nexti++)
			{
				delta+=((pArray->arr+i+1)->data+nexti)->delta*(*((*(pNeuralNetWork->net+i+1))->data+(pArray->arr+i)->size*nexti+index));
				//printf("%lf\n", *((*(pNeuralNetWork->net+i))->data+nexti+pVectorTrain->size*index));
			}
			(pVectorTrain->data+index)->delta = delta*(1.0-(pVectorTrain->data+index)->value)*(pVectorTrain->data+index)->value;
		}
		/*int nexti = 0;
		for(;nexti<(pArray->arr+i+1)->size;nexti++)
		{
			*((*(pNeuralNetWork->bias+i))->data+nexti) += -LNSTEP*(((pArray->arr+i+1)->data+nexti)->delta);
		}*/
	}

	for(i=pNeuralNetWork->hiddenCount-1;i>=0;i--)
	{
		int index = 0;
		VectorTrain* pVectorTrain= pArray->arr+i;
		for(;index < pVectorTrain->size; index++)
		{
			int nexti = 0;
			for(;nexti<(pVectorTrain+1)->size;nexti++)
			{
				*((*(pNeuralNetWork->net+i+1))->data+nexti*pVectorTrain->size+index) += -LNSTEP*(pVectorTrain->data+index)->value * (((pArray->arr+i+1)->data+nexti)->delta);
			}
		}
		int nexti = 0;
		for(;nexti<(pArray->arr+i+1)->size;nexti++)
		{
			*((*(pNeuralNetWork->bias+i))->data+nexti) += -LNSTEP*(((pArray->arr+i+1)->data+nexti)->delta);
		}

	}

	for(i=0;i<pNeuralNetWork->inputWidth;i++)
	{
		int j = 0;
		for(j=0;j<pArray->arr->size;j++)
		{
			*((*(pNeuralNetWork->net))->data+i+j*pNeuralNetWork->inputWidth) += -LNSTEP*(*(pInput->data+i))*((pArray->arr)->data+j)->delta;
		}
	}
}

void every_backward(NeuralNetWork* pNeuralNetWork, VectorTrainArray* pArray, Vector* pInput, Vector* pTarget, int layer, int nindex)
{
	int index = nindex;
	if(layer == pNeuralNetWork->hiddenCount)
	{
		VectorTrain* pOutputVectorTrain = pArray->arr+pNeuralNetWork->hiddenCount;
		int it = 0;
		for(;it<pNeuralNetWork->outWidth;it++)
		{
			double Ok = (pOutputVectorTrain->data+it)->value;
			double Tk = *(pTarget->data+it);
			(pOutputVectorTrain->data+it)->delta=(1.0-Ok)*Ok*(Ok-Tk);
		}
	}
	else if(layer != -1){

		int nexti = 0;
		if(nindex != pNeuralNetWork->hiddenWidth){
			int i = 0;
			VectorTrain* pVectorTrain= pArray->arr+layer;
			double delta = 0.0;
			for(;nexti<(pArray->arr+layer+1)->size;nexti++)
			{
				delta+=((pArray->arr+layer+1)->data+nexti)->delta*(*((*(pNeuralNetWork->net+layer+1))->data+(pArray->arr+layer)->size*nexti+index));
				*((*(pNeuralNetWork->net+layer+1))->data+nexti*pVectorTrain->size+index) += -LNSTEP*(pVectorTrain->data+index)->value * (((pArray->arr+layer+1)->data+nexti)->delta);
				//printf("%lf\n", *((*(pNeuralNetWork->net+i))->data+nexti+pVectorTrain->size*index));
			}
			(pVectorTrain->data+index)->delta = delta*(1.0-(pVectorTrain->data+index)->value)*(pVectorTrain->data+index)->value;
		}
		else
		{
			nexti = 0;
			for(;nexti<(pArray->arr+layer+1)->size;nexti++)
			{
				*((*(pNeuralNetWork->bias+layer))->data+nexti) += -LNSTEP*(((pArray->arr+layer+1)->data+nexti)->delta);
			}
		}
	}else
	{
		int j = 0;
		for(j=0;j<pArray->arr->size;j++)
		{
			*((*(pNeuralNetWork->net))->data+index+j*pNeuralNetWork->inputWidth) += -LNSTEP*(*(pInput->data+index))*((pArray->arr)->data+j)->delta;
		}
	}
}



void bki(Vector* lastdelta, NeuralNetWork* net, int layer)
{
	int i = 0;
	for(;i<lastdelta->size;i++)
	{
		//lastdelta->data[i]*net->data[i]
	}
}

void printVector(Vector* pVector)
{
	int i = 0;
	for (; i<pVector->size; i++)
	{
		printf("%.16lf\n", pVector->data[i]);
	}
}
double deltaO(double Ok, double Tk)
{
	return (1-Ok)*Ok*(Ok-Tk);
}
void calcmindelta()
{

}

/*
int main()
{
	NeuralNetWork* g_pNN = NULL;
	srand(time(NULL));
	if (!g_pNN)
		g_pNN = initNeuralNetwork(2, 5, 1000, 2);
	Vector* pIn = initVector(2);
	Vector* pTarget = initVector(2);
	pIn->data[0] = 0.0;
	pIn->data[1] = 1.0;
	pTarget->data[0] = 0.0;
	pTarget->data[1] = 1.0;
	int c = 1;
	int d = 1;
	int e = 1;
	//printVector(pIn);
	NeuralNetWork* pNeuralNetWork = g_pNN;
	VectorTrainArray* pOut = initVectorTrainArray(pNeuralNetWork->hiddenCount, pNeuralNetWork->hiddenWidth,pNeuralNetWork->outWidth);
	while (1) {
		*pIn->data = c;

		forward_train(g_pNN, pIn, pOut);
		backward(g_pNN, pOut, pIn, pTarget);
		int i = 0;
		double error = 0.0;
		for(;i<g_pNN->outWidth;i++)
		{
			error+=pow(pOut->arr[g_pNN->hiddenCount].data[i].value - pTarget->data[i], 2);
			printf("%.16lf %.16lf\n", pOut->arr[g_pNN->hiddenCount].data[i].value,pTarget->data[i]);
		}
		printf("%.16lf\n", error);
		//printVector(pOut);
		
		//releaseVector(pOut);
		//int ret = scanf("%d", &c);
	}
	releaseVector(pIn);
    return 0;
}
*/
