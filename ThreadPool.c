#include "ThreadPool.h"
#include "NNLab.h"
#include "stdio.h"

void * threadProc(void* ptr)
{
	PALPARAM* params = (PALPARAM*)ptr;
	while(1)
	{
	sem_wait(&params->evs);
	//pthread_mutex_lock(&params->m);

	if(params->type == FORWARD)
	{
		int nindex = params->nindex;
		params->nindex += 1;
		if(params->nindex<params->max_nindex)
			sem_post(&params->evs);
		//pthread_mutex_unlock(&params->m);
		every_forward_train(params->pNN, params->pRecord->pInput, params->pArray, params->layer, nindex);
		pthread_mutex_lock(&params->m);
		params->nindexundone -= 1;
		if(params->nindexundone == 0)
		{
			pthread_mutex_unlock(&params->m);
			sem_post(&params->evm);
		}
		else
		pthread_mutex_unlock(&params->m);
	}else

	if(params->type == BACKWARD)
	{
		int nindex = params->nindex;
		params->nindex += 1;
		if(params->nindex<params->max_nindex)
			sem_post(&params->evs);

		every_backward(params->pNN, params->pArray, params->pRecord->pInput, params->pRecord->pOutput, params->layer, nindex);
		pthread_mutex_lock(&params->m);
		params->nindexundone -= 1;
		if(params->nindexundone == 0)
		{
			pthread_mutex_unlock(&params->m);
			sem_post(&params->evm);
		}
		else
		pthread_mutex_unlock(&params->m);
	}

	}
}


void create_thread_pool(PALPARAM* params)
{
	int WORKER_COUNT = 8;
	int i = 0;
	pthread_t t;
	for(i=0;i<WORKER_COUNT;i++)
	{
		pthread_create(&t, NULL, threadProc, params);
	}
}


