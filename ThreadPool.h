#ifndef __THREADPOOL_H__
#define __THREADPOOL_H__
#include "NNLab.h"
#include "pthread.h"
#include <semaphore.h>
#define FORWARD 1
#define BACKWARD 2
typedef struct _palparam
{
	pthread_mutex_t m;
	sem_t evm;
	sem_t evs;
	NeuralNetWork* pNN;
	VectorTrainArray* pArray;
	Record* pRecord;
	int layer;
	int layerundone;
	int nindex;
	int max_nindex;
	int nindexundone;
	int type;
}PALPARAM;

void create_thread_pool(PALPARAM*);

#endif
