#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "unistd.h"
#include "time.h"

typedef struct chess {
	int chess[19][19];
	int turn;
}CHESS;

int init_chess(CHESS* pchess)
{
	pchess->turn = 1;
	int i = 0;
	for(;i<19;i++) {
		int j = 0;
		for(j=0;j<19;j++) {
			memset(&(pchess->chess[i][j]),0,sizeof(int));
		}
	}
}
int random_move(CHESS* pchess) {
	while(1) {
		int i = random()%19;
		int j = random()%19;
		if(pchess->chess[i][j] == 0) {
			pchess->chess[i][j] = pchess->turn;
			if(pchess->turn == 1)
				pchess->turn = 2;
			else
				pchess->turn = 1;
			break;
		}
	}
}

int printchess(CHESS* pchess)
{
	int i = 0;
	int j = 0;

	for(;i<19;i++) {
		for(j=0;j<19;j++) {
			printf(" %d  ",pchess->chess[i][j]);
		}
		printf("\n\n");
	}
}
int main() {
	srand(time(NULL));
	CHESS chess;
	init_chess(&chess);
	while(random_move(&chess)) {
		sleep(1);
		printchess(&chess);
		printf("\n\n\n\n\n");
	}
	return 0;
}

