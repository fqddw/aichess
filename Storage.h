#ifndef __STORAGE_H__
#define __STORAGE_H__
#include "NNLab.h"
#define TRUE 1L
int saveNeuralNetwork(NeuralNetWork*,char*);
NeuralNetWork* loadNeuralNetwork(char*);
#endif
