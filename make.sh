#/bin/sh
cc -O3 -o run chess.c train.c Storage.c NNLab.c NNsl.c -lm -lpthread
