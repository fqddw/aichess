#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "ctype.h"
#include "time.h"
#include "math.h"
#include "NNLab.h"
#include "NNsl.h"
#include "Storage.h"
#include "chess.h"
#define JU 0x1
#define MA (0x1<<1)
#define XIANG (0x1<<2)
#define SHI (0x1<<3)
#define JIANG (0x1<<4)
#define PAO (0x1<<5)
#define BING (0x1<<6)
#define MASK (0x1<<7)
#define BLACK MASK
#define RED 0
#define BOARD_SIZE 91
struct _move_list;
typedef struct _chess
{
	int chess[9][10];
	int turn;
}CHESS;

int getweightbyitem(int item)
{
	switch(item)
	{
		case JU:
			return 1000;
		case JU|MASK:
			return -1000;
		case MA:
			return 430;
		case MA|MASK:
			return -430;
		case PAO:
			return 450;
		case PAO|MASK:
			return -450;
		case XIANG:
			return 200;
		case XIANG|MASK:
			return -200;
		case SHI:
			return 200;
		case SHI|MASK:
			return -200;
		default:
			return 0;
	}
	return 0;
}
void init_weight(float*, int);
float calc_weight(float*,CHESS*);
float gweight[BOARD_SIZE];
int calvalue(CHESS* pchess)
{
	return 0;
	int count = 0;
	int i=0,j=0;
	int score = 0;
	for(i=0;i<9;i++)
	{
		for(j=0;j<10;j++)
		{
			score += getweightbyitem(pchess->chess[i][j]);
		}
	}
	return score;
}

typedef struct _move
{
	int sourcex;
	int sourcey;
	int destx;
	int desty;
	CHESS chess;
	struct _move_list* next;
}MOVE;
typedef struct _index
{
	int size;
	int index;
	int score;
}INDEX;

typedef struct _move_list
{
	int count;
	MOVE* move_list;
}MOVELIST;
typedef struct _treecoord
{
	int depth;
	INDEX* index;
}TREECOORD;
typedef struct _movetree
{
	MOVELIST* root;
}MOVETREE;
void printchessui(CHESS*);
void printchess(CHESS*);

double * chess_to_buffer(CHESS* pchess)
{
	double * weights_buffer = (double*)malloc((BOARD_SIZE+1)*sizeof(double));
	int index_x = 0;
	int index_y = 0;
	for(;index_x< 10; index_x++)
	{
		for(index_y = 0;index_y< 9; index_y++)
		{
			weights_buffer[index_x*9+index_y] = pchess->chess[index_y][index_x];
		}
	}
	weights_buffer[BOARD_SIZE-1] = pchess->turn;
	return weights_buffer;
}
/*
Record* chess_to_record(CHESS* pchess, int result)
{
	Record* pRecord = (Record*)malloc(sizeof(Record));
	pRecord->pInput = initVector(BOARD_SIZE);
	pRecord->pOutput = initVector(1);
	double * weights_buffer = pRecord->pInput->data;
	int index_x = 0;
	int index_y = 0;
	for(;index_x< 10; index_x++)
	{
		for(index_y = 0;index_y< 9; index_y++)
		{
			weights_buffer[index_x*9+index_y] = pchess->chess[index_y][index_x];
		}
	}
	weights_buffer[BOARD_SIZE-1] = pchess->turn;
	if(result == 1)
	{
		pRecord->pOutput->data[0] = 1.0;
		//pRecord->pOutput->data[1] = 0.0;
	}
	else
	{
		pRecord->pOutput->data[0] = 0.0;
		//pRecord->pOutput->data[1] = 1.0;
	}
	return pRecord;
}*/

CHESS* buffer_to_chess(double *weights_buffer)
{
	CHESS* pchess = (CHESS*)malloc(sizeof(CHESS));
	int index_x = 0;
	int index_y = 0;
	for (; index_x< 10; index_x++)
	{
		for (index_y = 0; index_y< 9; index_y++)
		{
			pchess->chess[index_y][index_x] = weights_buffer[index_x * 9 + index_y];
		}
	}
	pchess->turn = weights_buffer[BOARD_SIZE - 1];
	return pchess;
}
int printtreecoord(TREECOORD* treecoord)
{
	int i = 0;
	for(;i<treecoord->depth;i++)
	{
		printf("total depth %d depth %d index %d size %d\n",treecoord->depth,i,treecoord->index[i].index,treecoord->index[i].size);
	}
	return 0;
}

int invalid_move(int x,int y)
{
	if(x<9 && x>=0 && y>=0 && y<10)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
int canmove(CHESS* pchess,MOVE* move)
{
	return 0;
}
TREECOORD* init_treecoord(int depth)
{
	TREECOORD* treecoord = (TREECOORD*)malloc(sizeof(TREECOORD));
	treecoord->depth = depth;
	if(depth == 0)
	{
		treecoord->index = 0;
		return treecoord;
	}
	treecoord->index = (INDEX*)malloc(sizeof(INDEX)*depth);
	memset(treecoord->index,0,depth*sizeof(INDEX));
	return treecoord;
}

int getmovetreesize(MOVETREE* movetree,TREECOORD* treecoord)
{
	int i = 0;
	MOVELIST* movelist = movetree->root;
	for(;i<treecoord->depth;i++)
	{
		treecoord->index[i].size = movelist->count;
		movelist = movelist->move_list[treecoord->index[i].index].next;
	}
	return 0;
}
int incr(MOVETREE* movetree,TREECOORD* treecoord)
{
	int i = 0;
	//getmovetreesize(movetree,treecoord);
	if(treecoord->depth == 0)
		return 1;
	if(treecoord->index[treecoord->depth-1].index<treecoord->index[treecoord->depth-1].size - 1)
	{
		treecoord->index[treecoord->depth-1].index++;
		return 1;
	}
	else
	{
		for(;i<treecoord->depth;i++)
		{
			if(treecoord->index[treecoord->depth-i-1].index<treecoord->index[treecoord->depth-i-1].size - 1)
			{
				treecoord->index[treecoord->depth-i-1].index++;
				treecoord->index[treecoord->depth-i-1].size = 0;
				getmovetreesize(movetree,treecoord);
				return 1;
			}
			else
			{
				treecoord->index[treecoord->depth-i-1].index = 0;
			}
		}
	}
	return 0;
}

int is_end(TREECOORD* treecoord)
{
	int i = 0;
	for(i=0;i<treecoord->depth;i++)
	{
		if(treecoord->index[i].index !=treecoord->index[i].size - 1)
			return 0;
	}
	return 1;
}

int append_to_move_tree(MOVETREE* movetree,TREECOORD* treecoord,MOVELIST* lmovelist)
{
	int i = 0;
	MOVELIST* movelist = movetree->root;
	MOVELIST** needlist = 0;
	for(i=0;i<treecoord->depth;i++)
	{
		needlist = &(movelist->move_list+treecoord->index[i].index)->next;
		movelist = *needlist;
	}
	*needlist = lmovelist;
	return 0;
}
int clearchess(CHESS* pchess)
{
	free(pchess);
	return 0;
}
int copychess(CHESS* pchess,CHESS* newchess)
{
	newchess->turn =pchess->turn;
	int i = 9;
	int j = 10;
	for(i=0;i<9;i++)
	{
		for(j=0;j<10;j++)
		{
			newchess->chess[i][j] = pchess->chess[i][j];
		}
	}
	return 1;
}
int totalchesscount = 0;
int getchessbymove(CHESS* pchess,MOVE* move,CHESS* newchess)
{
	totalchesscount++;
	int i = move->destx;
	int j = move->desty;
	if(pchess->chess[move->sourcex][move->sourcey] == 0)
	{
		printchessui(pchess);
		printf("Invalid Move %d %d %d %d\n", move->sourcex, move->sourcey, i, j);
		exit(0);
		printf("Invalid Move\n");
	}
	copychess(pchess,newchess);
	newchess->chess[move->sourcex][move->sourcey] = 0;
	//if(pchess->chess[i][j] == 0)
	{
		newchess->chess[i][j] = pchess->chess[move->sourcex][move->sourcey];
	}
	newchess->turn = newchess->turn == BLACK?RED:BLACK;
	return 1;
}

int get_side_by_depth(int first_turn,int depth)
{
	int flag = depth%2;
	return (flag == 0)?(first_turn == BLACK?BLACK:RED):(first_turn == BLACK?RED:BLACK);
}

CHESS* getchessbytreecoord(CHESS* pchess,MOVETREE* movetree,TREECOORD* treecoord)
{
	if(!movetree->root)
	{
		return pchess;
	}
	int i = 0;
	MOVELIST* movelist = movetree->root;
	MOVE* nullmove = (MOVE*)0;
	MOVE tmpmove = {0};
	memset(&tmpmove,0,sizeof(MOVE));
	tmpmove.chess = *pchess;
	tmpmove.next = movelist;
	MOVE* move = &tmpmove;

	for(;i<treecoord->depth;i++)
	{
		move = move->next->move_list+treecoord->index[i].index;
	}
	return &move->chess;
}
int is_dead(CHESS* pchess,int side)
{
	return 0;
}
int append_movelist(MOVELIST* movelist,MOVE* move,CHESS* originchess)
{
	int count = movelist->count;
	movelist->count+=1;
	MOVE* new_movelist = (MOVE*)malloc(movelist->count*sizeof(MOVE));
	if(count!=0)
	{
		memcpy(new_movelist,movelist->move_list,count*sizeof(MOVE));
		free(movelist->move_list);
	}
	getchessbymove(originchess,move,&move->chess);
	calvalue(&move->chess);
	memcpy(new_movelist+count,move,sizeof(MOVE));
	movelist->move_list = new_movelist;
	return 1;
}
MOVELIST* get_move_list(CHESS* pchess)
{
	MOVELIST* movelist= (MOVELIST*)malloc(sizeof(MOVELIST));
	movelist->count = 0;
	movelist->move_list = 0;
	int i = 0;
	int j = 0;
	for(;i<10;i++)
	{
		for(j=0;j<9;j++)
		{
			int item = 0;
			if((pchess->chess[j][i] & MASK) != pchess->turn || pchess->chess[j][i] == 0)
			{
				continue;
			}

			switch(pchess->chess[j][i] &~MASK)
			{
				case JU:
					{
						int k = i+1;
						int l = j+1;
						for(;k<10;k++)
						{
							if(pchess->chess[j][k] != 0)
							{
								if((pchess->chess[j][k] & MASK) != pchess->turn)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = j;
									move.desty = k;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
								break;
							}
							else
							{
								MOVE move = {0};
								move.sourcex = j;
								move.sourcey = i;
								move.destx = j;
								move.desty = k;
								move.next = 0;
								append_movelist(movelist,&move,pchess);
							}
						}
						k = i-1;
						for(;k>=0;k--)
						{
							if(pchess->chess[j][k] != 0)
							{
								if((pchess->chess[j][k] & MASK) != pchess->turn)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = j;
									move.desty = k;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
								break;
							}
							else
							{
								MOVE move = {0};
								move.sourcex = j;
								move.sourcey = i;
								move.destx = j;
								move.desty = k;
								move.next = 0;
								append_movelist(movelist,&move,pchess);
							}
						}

						for(;l<9;l++)
						{
							if(pchess->chess[l][i] != 0)
							{
								if((pchess->chess[l][i] & MASK) != pchess->turn)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = l;
									move.desty = i;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
								break;
							}
							else
							{
								MOVE move = {0};
								move.sourcex = j;
								move.sourcey = i;
								move.destx = l;
								move.desty = i;
								move.next = 0;
								append_movelist(movelist,&move,pchess);
							}
						}
						l=j-1;
						for(;l>=0;l--)
						{
							if(pchess->chess[l][i] != 0)
							{
								if((pchess->chess[l][i] & MASK) != pchess->turn)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = l;
									move.desty = i;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
								break;
							}
							else
							{
								MOVE move = {0};
								move.sourcex = j;
								move.sourcey = i;
								move.destx = l;
								move.desty = i;
								move.next = 0;
								append_movelist(movelist,&move,pchess);
							}
						}

					}
					break;
				case MA:
					{
						MOVE lefttop,leftbottom,topleft,topright,righttop,rightbottom,bottomleft,bottomright;
						MOVE move;
						move.sourcex = j;
						move.sourcey = i;
						move.destx = j - 2;
						move.desty = i - 1;
						lefttop = move;
						if(invalid_move(j-2,i-1))
						{
							if(!pchess->chess[j-1][i])
							{
								if(pchess->chess[j-2][i-1] == 0 || (pchess->chess[j-2][i-1] & MASK) != pchess->turn)
								{
									append_movelist(movelist,&lefttop,pchess);
								}
							}
						}
						move.destx = j - 1;
						move.desty = i - 2;
						topleft = move;
						if(invalid_move(j-1,i-2))
							if(!pchess->chess[j][i-1])
								if(pchess->chess[j-1][i-2] == 0 || (pchess->chess[j-1][i-2] & MASK) != pchess->turn)
									append_movelist(movelist,&topleft,pchess);

						move.destx = j - 2;
						move.desty = i + 1;
						leftbottom = move;
						if(invalid_move(j-2,i+1))
							if(!pchess->chess[j-1][i])
								if(pchess->chess[j-2][i+1] ==0 || (pchess->chess[j-2][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&leftbottom,pchess);

						move.destx = j - 1;
						move.desty = i + 2;
						bottomleft = move;
						if(invalid_move(j-1,i+2))
						{
							if(!pchess->chess[j][i+1])
							{
								if(pchess->chess[j-1][i+2] ==0 || (pchess->chess[j-1][i+2] & MASK) != pchess->turn)
								{
									append_movelist(movelist,&bottomleft,pchess);
								}
							}
						}

						move.destx = j + 1;
						move.desty = i + 2;
						bottomright = move;
						if(invalid_move(j+1,i+2))
							if(!pchess->chess[j][i+1])
								if(pchess->chess[j+1][i+2] == 0 || (pchess->chess[j+1][i+2] & MASK) != pchess->turn)
									append_movelist(movelist,&bottomright,pchess);
						move.destx = j + 2;
						move.desty = i + 1;
						rightbottom = move;
						if(invalid_move(j+2,i+1))
							if(!pchess->chess[j+1][i])
								if(pchess->chess[j+2][i+1] == 0 || (pchess->chess[j+2][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&rightbottom,pchess);
						move.destx = j + 2;
						move.desty = i - 1;
						righttop = move;
						if(invalid_move(j+2,i-1))
							if(!pchess->chess[j+1][i])
								if(pchess->chess[j+2][i-1] == 0 || (pchess->chess[j+2][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&righttop,pchess);					

						move.destx = j + 1;
						move.desty = i - 2;
						topright = move;
						if(invalid_move(j+1,i-2))
							if(!pchess->chess[j][i-1])
								if(pchess->chess[j+1][i-2] == 0 || (pchess->chess[j+1][i-2] & MASK) != pchess->turn)
									append_movelist(movelist,&topright,pchess);
					}
					break;
				case PAO:
					{
						int k = i+1;
						int l = j + 1;
						int flag = 1;
						for(;k<10;k++)
						{
							if(pchess->chess[j][k] != 0)
							{
								if(flag == 2)
								{
									if(pchess->turn == (pchess->chess[j][k]&MASK))
									{
										break;
									}
									else
									{
										MOVE move = {0};
										move.sourcex = j;
										move.sourcey = i;
										move.destx = j;
										move.desty = k;
										move.next = 0;
										append_movelist(movelist,&move,pchess);
										break;
									}
								}
								else
									flag = 2;
							}
							else
							{
								if(flag == 1)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = j;
									move.desty = k;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
							}
						}
						k = i-1;
						flag = 1;
						for(;k>=0;k--)
						{
							if(pchess->chess[j][k] != 0)
							{
								if(flag == 2)
								{
									if(pchess->turn == (pchess->chess[j][k]&MASK))
									{
										break;
									}
									else
									{
										MOVE move = {0};
										move.sourcex = j;
										move.sourcey = i;
										move.destx = j;
										move.desty = k;
										move.next = 0;
										append_movelist(movelist,&move,pchess);
										break;
									}
								}
								else
									flag = 2;
							}
							else
							{
								if(flag == 1)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = j;
									move.desty = k;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
							}
						}
						l = j+1;
						flag = 1;
						for(;l<9;l++)
						{
							if(pchess->chess[l][i] != 0)
							{
								if(flag == 2)
								{
									if(pchess->turn == (pchess->chess[l][i]&MASK))
									{
										break;
									}
									else
									{
										MOVE move = {0};
										move.sourcex = j;
										move.sourcey = i;
										move.destx = l;
										move.desty = i;
										move.next = 0;
										append_movelist(movelist,&move,pchess);
										break;
									}
								}
								else
									flag = 2;
							}
							else
							{
								if(flag == 1)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = l;
									move.desty = i;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
							}
						}
						l = j-1;
						flag = 1;
						for(;l >= 0;l--)
						{
							if(pchess->chess[l][i] != 0)
							{
								if(flag == 2)
								{
									if(pchess->turn == (pchess->chess[l][i]&MASK))
									{
										break;
									}
									else
									{
										MOVE move = {0};
										move.sourcex = j;
										move.sourcey = i;
										move.destx = l;
										move.desty = i;
										move.next = 0;
										append_movelist(movelist,&move,pchess);
										break;
									}
								}
								else
									flag = 2;
							}
							else
							{
								if(flag == 1)
								{
									MOVE move = {0};
									move.sourcex = j;
									move.sourcey = i;
									move.destx = l;
									move.desty = i;
									move.next = 0;
									append_movelist(movelist,&move,pchess);
								}
							}
						}

					}
					break;
				case XIANG:
					{
						MOVE lefttop,leftbottom,righttop,rightbottom;
						lefttop.sourcex = j;
						lefttop.sourcey = i;
						lefttop.desty = i-2;
						lefttop.destx = j-2;
						if(invalid_move(j-2,i-2))
						{
							if(!((pchess->turn == BLACK) && (i-2<4))) {
								if(pchess->chess[j-2][i-2] == 0 || (pchess->chess[j-2][i-2] & MASK) != pchess->turn)
									if (pchess->chess[j - 1][i - 1] == 0)
										append_movelist(movelist,&lefttop,pchess);
							}
						}
						leftbottom.sourcex = j;
						leftbottom.sourcey = i;
						leftbottom.destx = j - 2;
						leftbottom.desty = i + 2;
						if(invalid_move(j-2,i+2))
						{
							if(!((pchess->turn == RED) && (i+2>5))) {
								if(pchess->chess[j-2][i+2] == 0 || (pchess->chess[j-2][i+2] & MASK) != pchess->turn)
									if (pchess->chess[j - 1][i + 1] == 0)
										append_movelist(movelist,&leftbottom,pchess);
							}
						}
						rightbottom.sourcex = j;
						rightbottom.sourcey = i;
						rightbottom.destx = j + 2;
						rightbottom.desty = i + 2;
						if(invalid_move(j+2,i+2))
						{
							if(!((pchess->turn == RED) && (i+2>5))) {
								if(pchess->chess[j+2][i+2] == 0 || (pchess->chess[j+2][i+2] & MASK) != pchess->turn)
									if (pchess->chess[j + 1][i + 1] == 0)
										append_movelist(movelist,&rightbottom,pchess);
							}
						}
						righttop.sourcex = j;
						righttop.sourcey = i;
						righttop.destx = j + 2;
						righttop.desty = i - 2;
						if(invalid_move(j+2,i-2))
						{
							if(!((pchess->turn == BLACK) && (i-2<4))) {
								if(pchess->chess[j+2][i-2] == 0 || (pchess->chess[j+2][i-2] & MASK) != pchess->turn)
									if (pchess->chess[j + 1][i - 1] == 0)
										append_movelist(movelist,&righttop,pchess);
							}
						}

					}
					break;
				case SHI:
					{
						MOVE lefttop,leftbottom,righttop,rightbottom;
						lefttop.sourcex = j;
						lefttop.sourcey = i;
						lefttop.destx = j-1;
						lefttop.desty = i-1;
						if(pchess->turn == BLACK)
						{
							if(i-1>=7 && i-1<=9 && j-1 >=3 && j-1<=5)
							{
								if(pchess->chess[j-1][i-1] == 0 || (pchess->chess[j-1][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&lefttop,pchess);
							}
						}

						if(pchess->turn == RED)
						{
							if(i-1>=0 && i-1<=2 && j-1 >=3 && j-1<=5)
							{
								if(pchess->chess[j-1][i-1] == 0 || (pchess->chess[j-1][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&lefttop,pchess);
							}
						}
						leftbottom.sourcex = j;
						leftbottom.sourcey = i;
						leftbottom.destx = j-1;
						leftbottom.desty = i+1;
						if(pchess->turn == BLACK)
						{
							if(i+1>=7 && i+1<=9 && j-1 >=3 && j-1<=5)
							{
								if(pchess->chess[j-1][i+1] == 0 || (pchess->chess[j-1][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&leftbottom,pchess);
							}
						}

						if(pchess->turn == RED)
						{
							if(i+1>=0 && i+1<=2 && j-1 >=3 && j-1<=5)
							{
								if(pchess->chess[j-1][i+1] == 0 || (pchess->chess[j-1][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&leftbottom,pchess);
							}
						}
						righttop.sourcex = j;
						righttop.sourcey = i;
						righttop.destx = j+1;
						righttop.desty = i-1;
						if(pchess->turn == BLACK)
						{
							if(i-1>=7 && i-1<=9 && j+1 >=3 && j+1<=5)
							{
								if(pchess->chess[j+1][i-1] == 0 || (pchess->chess[j+1][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&righttop,pchess);
							}
						}

						if(pchess->turn == RED)
						{
							if(i-1>=0 && i-1<=2 && j+1 >=3 && j+1<=5)
							{
								if(pchess->chess[j+1][i-1] == 0 || (pchess->chess[j+1][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&righttop,pchess);
							}
						}
						rightbottom.sourcex = j;
						rightbottom.sourcey = i;
						rightbottom.destx = j+1;
						rightbottom.desty = i+1;
						if(pchess->turn == BLACK)
						{
							if(i+1>=7 && i+1<=9 && j+1 >=3 && j+1<=5)
							{
								if(pchess->chess[j+1][i+1] == 0 || (pchess->chess[j+1][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&rightbottom,pchess);
							}
						}

						if(pchess->turn == RED)
						{
							if(i+1>=0 && i+1<=2 && j+1 >=3 && j+1<=5)
							{
								if(pchess->chess[j+1][i+1] == 0 || (pchess->chess[j+1][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&rightbottom,pchess);
							}
						}

					}
					break;
				case JIANG:
					{
						MOVE left,right,top,bottom;
						top.sourcex = j;
						top.sourcey = i;
						top.destx = j;
						top.desty = i-1;
						if(pchess->turn == RED)
						{
							if(i-1>=0)
								if (pchess->chess[j][i - 1] == 0 || (pchess->chess[j][i - 1] & MASK) != pchess->turn)
								{
									append_movelist(movelist, &top, pchess);
								}
						}
						if(pchess->turn == BLACK)
						{
							if(i-1>=7)
								if(pchess->chess[j][i-1] == 0 || (pchess->chess[j][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&top,pchess);
						}

						bottom.sourcex = j;
						bottom.sourcey = i;
						bottom.destx = j;
						bottom.desty = i+1;
						if(pchess->turn == RED)
						{
							if(i+1<=2)
								if(pchess->chess[j][i+1] == 0 || (pchess->chess[j][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&bottom,pchess);
						}
						if(pchess->turn == BLACK)
						{
							if(i+1>=7 && i+1<=9)
								if(pchess->chess[j][i+1] == 0 || (pchess->chess[j][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&bottom,pchess);
						}
						left.sourcex = j;
						left.sourcey = i;
						left.destx = j-1;
						left.desty = i;
						if(j-1>=3)
							if(pchess->chess[j-1][i] == 0 || (pchess->chess[j-1][i] & MASK) != pchess->turn)
								append_movelist(movelist,&left,pchess);
						right.sourcex = j;
						right.sourcey = i;
						right.destx = j+1;
						right.desty = i;
						if(j+1<=5)
							if(pchess->chess[j+1][i] == 0 || (pchess->chess[j+1][i] & MASK) != pchess->turn)
								append_movelist(movelist,&right,pchess);


					}
					break;
				case BING:
					{
						MOVE left,right,front;
						if((pchess->turn == BLACK && i<5) ||(pchess->turn == RED && i>4))
						{
							left.sourcex = j;
							left.sourcey = i;
							left.destx = j - 1;
							left.desty = i;
							if(j-1>=0)
								if(pchess->chess[j-1][i] == 0 || (pchess->chess[j-1][i] & MASK) != pchess->turn)
									append_movelist(movelist,&left,pchess);
							right.sourcex = j;
							right.sourcey = i;
							right.destx = j + 1;
							right.desty = i;
							if(j+1<9)
								if(pchess->chess[j+1][i] == 0 || (pchess->chess[j+1][i] & MASK) != pchess->turn)
									append_movelist(movelist,&right,pchess);

						}

						if(pchess->turn == BLACK)
						{
							front.sourcex = j;
							front.sourcey = i;
							front.destx = j;
							front.desty = i-1;
							if(i-1>=0)
								if(pchess->chess[j][i-1] == 0 || (pchess->chess[j][i-1] & MASK) != pchess->turn)
									append_movelist(movelist,&front,pchess);
						}

						if(pchess->turn == RED)
						{
							front.sourcex = j;
							front.sourcey = i;
							front.destx = j;
							front.desty = i+1;
							if(i+1<10)
								if(pchess->chess[j][i+1] == 0 || (pchess->chess[j][i+1] & MASK) != pchess->turn)
									append_movelist(movelist,&front,pchess);
						}
					}
					break;
				default:
					;
			}
		}
	}
	return movelist;
}
int cleantreecoord(TREECOORD* treecoord)
{
	if(treecoord->index)
		free(treecoord->index);
	free(treecoord);
	return 0;
}
int total = 0;
TREECOORD * bestmove;
MOVELIST* nextmove(CHESS* pchess)
{	
	int count = 0;
	TREECOORD* treecoord = init_treecoord(0);
	bestmove = init_treecoord(0);
	MOVETREE* movetree = (MOVETREE*)malloc(sizeof(MOVETREE));
	movetree->root = NULL;
	CHESS* pchesscur = pchess;
	do
	{
		if(treecoord->depth>0)
			if(treecoord->index[treecoord->depth-1].index == 0) {
				pchesscur = getchessbytreecoord(pchess,movetree,treecoord);
			}
		MOVELIST* list_ptr = get_move_list(pchesscur);
		if(list_ptr->count == 0)
		{
			printf("---------------\n");
			return 0;
		}
		int i = 0;
		total+=list_ptr->count;
		MOVELIST* available_list = list_ptr;/*(MOVELIST*)malloc(sizeof(MOVELIST));
						      available_list->count = 0;
						      available_list->move_list = 0;
						      for(;i<list_ptr->count;i++)
						      {
						      MOVE* move = list_ptr->move_list+i;
						      CHESS newchess = {0};
						      CHESS* pnewchess = &newchess;
						      getchessbymove(pchesscur,move,pnewchess);
						      int score = calvalue(pnewchess);
						      if(treecoord->depth>1)
						      if(pchesscur->turn == RED)
						      {
						      if(score > bestmove->index[treecoord->depth-1].score)
						      {
						      bestmove->index[treecoord->depth-1].score = score;
						      }
						      }

						      if(treecoord->depth>1)
						      if(pchesscur->turn == BLACK)
						      {
						      if(score < bestmove->index[treecoord->depth-1].score)
						      {
						      bestmove->index[treecoord->depth-1].score = score;
						      }
						      }

						      if(score > -2000)
						      {
						      append_movelist(available_list,move);
						      }
						      }
						      free(list_ptr->move_list);
						      free(list_ptr);*/
		if(!movetree->root)
			movetree->root = available_list;
		else
		{
			append_to_move_tree(movetree,treecoord,available_list);
		}
		if(is_end(treecoord))
		{
			TREECOORD* newtreecoord = init_treecoord(treecoord->depth+1);
			TREECOORD* tmpbestmove = init_treecoord(treecoord->depth+1);
			memcpy(tmpbestmove->index,bestmove->index,treecoord->depth*sizeof(INDEX));
			cleantreecoord(bestmove);
			bestmove = tmpbestmove;
			cleantreecoord(treecoord);
			treecoord = newtreecoord;
			getmovetreesize(movetree,treecoord);
			count++;
			printf("TotalCount %d %d %d %d\n",treecoord->depth-1,count,totalchesscount,total);
			count = 0;
			if(treecoord->depth == 5)
				exit(0);
		}
		else
		{
			incr(movetree,treecoord);
			count++;
		}
	}while(1);
}

int getchesscode(char code)
{
	switch(code)
	{
		case 'r':
			return JU | BLACK;
		case 'n':
			return MA | BLACK;
		case 'b':
			return XIANG | BLACK;
		case 'a':
			return SHI | BLACK;
		case 'k':
			return JIANG | BLACK;
		case 'c':
			return PAO | BLACK;
		case 'p':
			return BING | BLACK;

		case 'R':
			return JU | RED;
		case 'N':
			return MA | RED;
		case 'B':
			return XIANG | RED;
		case 'A':
			return SHI | RED;
		case 'K':
			return JIANG | RED;
		case 'C':
			return PAO | RED;
		case 'P':
			return BING | RED;

		default:
			return 0;
	}
}

CHESS* get_chess_from_fen(char* fen)
{
	int line = 0;
	CHESS* pchess = (CHESS*)malloc(sizeof(CHESS));
	pchess->turn = RED;
	int i = 0;
	int j = 0;
	for(i=0;i<9;i++)
	{
		for(j=0;j<10;j++)
		{
			pchess->chess[i][j] = 0;
		}
	}
	int flag = 0;
	char* data = fen;
	while(line<10)
	{
		int i = 0;
		int chessoffset = 0;
		for(;;i++)
		{
			if(data[i] == ' ') {
				if(data[i+1] == '1')
					pchess->turn = RED;
				else
					pchess->turn = BLACK;
				return pchess;
			}
			if(data[i] == '/')
			{
				data = data+i+1;
				line++;
				break;
			}
			else
			{
				if(isdigit(data[i]))
				{
					int it = 0;

					int len = data[i] - '0';
					for(;it<len;it++)
					{
						pchess->chess[i+it][line] = 0;
					}
					chessoffset += len-1;
				}
				else
				{
					int code = getchesscode(data[i]);
					pchess->chess[i+chessoffset][line] = code;
				}
			}
		}
	}
	return pchess;
}
void printchess(CHESS* pchess)
{
	int i = 0;
	int j = 0;
	for(j=0;j<10;j++)
	{
		for(i=0;i<9;i++)
		{
			printf("%5d ",pchess->chess[i][j]);
		}
		printf("\n");
	}
}

void printchessui(CHESS* pchess)
{
	int i = 0;
	int j = 0;
	char* str = NULL;
	for(j=0;j<10;j++)
	{
		for(i=0;i<9;i++)
		{
			if(pchess->chess[i][j] == JU)
				str = "\033[40;41m 車\033[0m ";
			else if(pchess->chess[i][j] == (JU|MASK))
				str = "\033[40;37m 車\033[0m ";
			else if(pchess->chess[i][j] == MA)
				str = "\033[40;41m 马\033[0m ";
			else if(pchess->chess[i][j] == (MA|MASK))
				str = "\033[40;37m 马\033[0m ";
			else if(pchess->chess[i][j] == XIANG)
				str = "\033[40;41m 相\033[0m ";
			else if(pchess->chess[i][j] == (XIANG|MASK))
				str = "\033[40;37m 象\033[0m ";
			else if(pchess->chess[i][j] == SHI)
				str = "\033[40;41m 士\033[0m ";
			else if(pchess->chess[i][j] == (SHI|MASK))
				str = "\033[40;37m 士\033[0m ";
			else if(pchess->chess[i][j] == JIANG)
				str = "\033[40;41m 帅\033[0m ";
			else if(pchess->chess[i][j] == (JIANG|MASK))
				str = "\033[40;37m 将\033[0m ";
			else if(pchess->chess[i][j] == BING)
				str = "\033[40;41m 兵\033[0m ";
			else if(pchess->chess[i][j] == (BING|MASK))
				str = "\033[40;37m 卒\033[0m ";
			else if(pchess->chess[i][j] == PAO)
				str = "\033[40;41m 炮\033[0m ";
			else if(pchess->chess[i][j] == (PAO|MASK))
				str = "\033[40;37m 炮\033[0m ";

			else
				str = "    ";

			printf("%s", str);
		}
		printf("\n");
	}
}

char* to_fen_string(CHESS* pchess)
{
	int i = 0;
	int j = 0;
	for(;j<10;j++)
	{
		int blank = 0;
		for(i=0;i<9;i++)
		{
			if(pchess->chess[i][j] == 0)
			{
				blank++;
			}
			else
			{
			}
		}
	}
	return 0;
}
int countrev = 0;
int depthrev = 5;
int nextmoverev(CHESS* pchess)
{
	total++;
	if(total>depthrev)
		return 0;
	countrev++;
	/*	if(total == 1)
		depthrev++;
		*/
	CHESS* pchesscur = pchess;
	MOVELIST* movelist = get_move_list(pchesscur);
	int i = 0;
	for(i=0;i<movelist->count;i++)
	{
		CHESS chessnow = {0};
		CHESS* pchessnow = &chessnow;
		getchessbymove(pchesscur,movelist->move_list+i,pchessnow);
		if(calvalue(pchessnow)>-2000)
		{
		}
		nextmoverev(pchessnow);
		total--;
	}
	return 0;
}

int check_end(CHESS* pchess)
{
	int i = 0;
	int j = 0;
	int irjiang = 0;
	int jrjiang = 0;
	int ibjiang = 0;
	int jbjiang = 0;
	int rflag = 0;
	int bflag = 0;
	for(;i<9;i++)
	{
		for(j=0;j<10;j++)
		{
			if(pchess->chess[i][j] == JIANG)
			{
				irjiang = i;
				jrjiang = j;
				rflag = 1;
			}
			if(pchess->chess[i][j] == (JIANG|BLACK))
			{
				ibjiang = i;
				jbjiang = j;
				bflag = 1;
			}
		}
	}
	if(bflag && rflag)
	{
		if( irjiang == ibjiang)
		{
			int emptylineflag = 0;
			int it = jrjiang+1;
			for(;it<jbjiang;it++)
			{
				if(pchess->chess[irjiang][it] != 0)
				{
					emptylineflag = 1;
					break;
				}
			}
			if(emptylineflag == 0)
			{
				if(pchess->turn == RED)
				{
					return 1;
				}
				if(pchess->turn == BLACK)
				{
					return 2;
				}
			}
		}
		return 0;
	}
	else
	{
		if(bflag)
			return 2;
		if(rflag)
			return 1;
		return 0;
	}
}

typedef struct _node
{
	CHESS* pchess;
	struct _node* next;
}NODE;
typedef struct _list
{
	NODE* head;
}LIST;

LIST* alloc_list()
{
	LIST* plist = (LIST*)malloc(sizeof(LIST));
	plist->head = NULL;
	return plist;
}

NODE* alloc_node()
{
	NODE* node = (NODE*)malloc(sizeof(NODE));
	node->pchess = NULL;
	node->next = NULL;
	return node;
}
void append_list(LIST* plist, CHESS* pchess)
{
	if(plist->head == NULL)
	{
		NODE* node = alloc_node();
		node->pchess = pchess;
		node->next = NULL;
	}
	else
	{
	}
}

void save_vs_chess()
{
	CHESS* pchessbk = get_chess_from_fen("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr 1");
}

double* get_random_vs(CHESS* pchessin)
{
	CHESS* pchess = pchessin;
	CHESS* pchessbk = pchess;
	int round = 0;
	int rwin = 0;
	int bwin = 0;
	int i = 0;
	FILE* fp = fopen("cr.qp", "ab");
	MOVELIST* rml = get_move_list(pchess);
	while(round < 1000)
	{
		MOVELIST* prev = NULL;
		int index = 0;
		while(pchess)
		{
			MOVELIST* ml = get_move_list(pchess);


			int count = ml->count;
			int offset = 0;
			int bestmove = 0;
			float max = 0;
			/*if(pchess->turn == RED)
			  {
			  for(;bestmove < count; bestmove++) {
			  float cur_weight = calc_weight(gweight, &((ml->move_list+bestmove)->chess));
			  if(cur_weight >= max) {
			  max = cur_weight;
			  offset = bestmove;
			  }
			  }
			  }*/
			/*else if (pchess->turn == BLACK)
			  {*/
			offset = rand() % count;
			/*}*/
			//printf("%d %d %d\n", offset, count, pchess->turn);
			//printf("Best Move offset %d maxweight %f\n", offset, max);
			//printf("count random %d %d\n",count,random);
			MOVE* mv = ml->move_list+offset;
			int ckret = 0;
			/*{
			  int mvindex = 0;
			  for(mvindex = 0;mvindex<count;mvindex++)
			  {
			  CHESS* ckchess = &(ml->move_list+mvindex)->chess;
			  ckret = 0;
			  ckret = check_end(ckchess);
			  if(ckret != 0)
			  {
			  exit(0);
			  break;
			  }
			  }
			  }*/
			pchess = &mv->chess;
			//printf("chess %d %d\n", index, offset);
			//printchess(pchess);
			index++;
			int ret = check_end(pchess);
			//printf("%d\n",ret);
			if(index>2000)
			{
				ret = 3;
				printf("step out\n");
			}
			if(ckret != 0)
				ret = ckret;
			/*if(ret<0)
			{
				free(ml->move_list);
				free(ml);
				pchess = pchessbk;
				continue;
			}*/
			if(ret == 1)
			{
				rwin++;
				//printchess(pchess);
				//printf("R Win %d\n",index);
				//fwrite(chess_to_buffer(pchess), BOARD_SIZE*sizeof(int),1, stdout);
				double* buffer = chess_to_buffer(pchess);
				buffer[BOARD_SIZE] = 1.0;
				fwrite(buffer, sizeof(double)*(BOARD_SIZE+1), 1, fp);
				fflush(fp);
				free(ml->move_list);
				free(ml);
				//return buffer;
				float weight_result = 0.0;
				//weight_result = calc_weight(gweight, pchess);
				//printchess(pchess);
				//printf("R win Weight %f\n", weight_result);
				pchess = pchessbk;
				break;
			}
			if(ret == 2)
			{
				bwin++;
				//printchess(pchess);
				//printf("B Win %d\n",index);
				//fwrite(chess_to_buffer(pchess), BOARD_SIZE*sizeof(int), 1, stdout);
				double* buffer = chess_to_buffer(pchess);
				buffer[BOARD_SIZE] = 0;
				fwrite(buffer, sizeof(double)*(BOARD_SIZE+1), 1, fp);
				fflush(fp);

				free(ml->move_list);
				free(ml);

				//return buffer;
				float weight_result = 0.0;
				//weight_result = calc_weight(gweight, pchess);
				//printchess(pchess);
				//printf("B win Weight %f\n", weight_result);
				pchess = pchessbk;
				break;
			}
			if(ret == 3)
			{
				//printf("Out Of Step\n");
				free(ml->move_list);
				free(ml);
				pchess = pchessbk;
				//return NULL;
				break;
			}
			if(ret == 4)
			{
				pchess = pchessbk;
				break;
			}
			if (prev != NULL){
				free(prev->move_list);
				free(prev);
			}
			prev = ml;
		}
		round++;
		//printf("Round %d\n",round);
	}
	//printf("rwin %d bwin %d\n",rwin,bwin);
	return 0;
}
void init_weight(float* weight,int size) {
	int i = 0;
	for(i=0;i<size;i++) {
		weight[i] = rand() % 1000;
	}
}

float calc_weight(float* weight, CHESS* pchess) {
	int i = 0;
	int j = 0;
	float result = 0;
	for(i=0;i<9;i++)
	{
		for(j=0;j<10;j++) {
			/*result += weight[i+j*9]*pchess->chess[i][j];*/
			if (pchess->chess[i][j] != 0)
			{
				if ((pchess->chess[i][j] & MASK) == RED)
				{
					result += pchess->chess[i][j]+j;
				}
				else
				{
					result -= pchess->chess[i][j]+10-j;
				}
			}
		}
	}
	//result += weight[BOARD_SIZE-1]*pchess->turn;
	return result;
}

int addtomatrix(double** matrix, int width, int *height, int* log, int* flag, double* buffer)
{
	int finish = 0;
	while(!finish)
	{
		int j=0;
		for(;j<width;j++)
		{
			printf("%lf ",buffer[j]);
		}
		printf("\n");

		int firstnonezero = 0;
		int i=0;
		for(i=0;i<width;i++)
		{
			if(buffer[i] != 0)
			{
				//printf("none zero %d %lf\n", i, buffer[i]);
				firstnonezero = i;
				break;
			}
		}

		if(flag[firstnonezero] != 1)
		{
			flag[firstnonezero] = 1;
			log[firstnonezero] = *height;
			matrix[*height] = buffer;
			(*height)++;
			finish = 1;
		}
		else
		{
			int allzero = 0;
			double firstnonezeroval = buffer[firstnonezero];
			for(i=0;i<width;i++)
			{
				if(i == firstnonezero)
					buffer[i] = 0;
				else
				{
					buffer[i]-=(buffer[i]*firstnonezeroval/matrix[log[firstnonezero]][firstnonezero]);
					//printf("%lf %d %lf %d %lf\n", buffer[i], i, firstnonezeroval, firstnonezero, matrix[log[firstnonezero]][firstnonezero]);
				}
				if(flag[BOARD_SIZE] == 1)
				{
					//printf(" %lf", matrix[log[BOARD_SIZE]][i]);
				}
				//printf(" %d", flag[i]);
				if(buffer[i] != 0)
					allzero = 1;
			}
			if(allzero == 0)
			{
				return 0;
			}
		}
	}
	return 0;
}

int simulate(CHESS* pchess)
{
	CHESS* pcur_chess = pchess;
	MOVELIST* tmpml = NULL;
	while (1)
	{
		MOVELIST* plist = get_move_list(pcur_chess);
		int r = rand() % plist->count;
		pcur_chess = &(plist->move_list[r].chess);
		printchessui(pcur_chess);
		//printchess(pcur_chess);
		int res = check_end(pcur_chess);
		if (tmpml)
		{
			free(tmpml->move_list);
			free(tmpml);
		}
		tmpml = plist;

		if (res)
		{
			free(plist->move_list);
			free(plist);
			return res;
		} 
	}
	return 0;
}

void chesstobuffer(CHESS* pchess, int* pchessbuffer)
{
	int i=0;
	int j=0;
	for(i=0;i<10;i++)
	{
		for(j=0;j<9;j++)
		{
			pchessbuffer[j+i*9] = pchess->chess[j][i];
		}
	}

}
int dot_chain(CHESS* pchess)
{
	FILE* fp = fopen("chesslog.bin", "ab+");
	int pchessbuffer[91] = {0};
	chesstobuffer(pchess,pchessbuffer);
	fwrite(pchessbuffer, 91*sizeof(int), 1, fp);
	CHESS* pcur_chess = pchess;
	MOVELIST* tmpml = NULL;
	while (1)
	{
		MOVELIST* plist = get_move_list(pcur_chess);
		int r = rand() % plist->count;
		pcur_chess = &(plist->move_list[r].chess);
		chesstobuffer(pcur_chess,pchessbuffer);
		//fwrite(pchessbuffer, 91*sizeof(int), 1, fp);
		//printchess(pcur_chess);
		int res = check_end(pcur_chess);
		if (tmpml)
		{
			free(tmpml->move_list);
			free(tmpml);
		}
		tmpml = plist;

		if (res)
		{
			if(res == 1)
			{
				int rwin = -1;
				fwrite(&rwin, sizeof(int), 1, fp);
			}
			if(res == 1)
			{
				int bwin = -2;
				fwrite(&bwin, sizeof(int), 1, fp);
			}
			fclose(fp);
			free(plist->move_list);
			free(plist);
			return res;
		} 
	}
	return 0;
}

typedef struct _WIN_RATE
{
	int rwin;
	int count;
	struct _WIN_RATE* parent;
	struct _WIN_RATE* children;
}WIN_RATE;

double calc_ucb_value(WIN_RATE* wr)
{
	double val = (double)wr->rwin / wr->count + sqrt(2*log((double)wr->parent->count)/ wr->count);
	//printf(" %lf\n", val);
	return val;
}
int backward_m(WIN_RATE* wr, TREECOORD* treecoord, int winres, int turn)
{
	int i=0;
	if((winres == 1 && turn == RED) ||(winres == 2 && turn == BLACK))
		wr->parent->rwin += 1;
	wr->parent->count += 1;
	for(;i<treecoord->depth;i++)
	{
		if((winres == 1 && turn == RED) ||(winres == 2 && turn == BLACK))
			(wr+treecoord->index[i].index)->rwin += 1;
		(wr+treecoord->index[i].index)->count += 1;
		wr = (wr+treecoord->index[i].index)->children;
	}
	return 0;
}
MOVE* getmovebytreecoord(MOVETREE* mt, TREECOORD* treecoord)
{
	int i = 0;
	MOVE* move;
	MOVELIST* ml = mt->root;
	MOVELIST* tmp = mt->root;
	for(;i<treecoord->depth;i++)
	{
		move = tmp->move_list+treecoord->index[i].index;
		tmp = move->next;
	}
	return move;
}
WIN_RATE* getwinratebytreecoord(WIN_RATE* wr, TREECOORD* treecoord)
{
	int i = 0;
	WIN_RATE* winrate_res;
	WIN_RATE* tmp = wr;
	for(;i<treecoord->depth;i++)
	{
		winrate_res = tmp+treecoord->index[i].index;
		tmp = winrate_res->children;
	}
	return winrate_res;
}

int expansion(WIN_RATE* win_rates, MOVETREE* mt, TREECOORD* treecoord)
{
	MOVE* pmove = getmovebytreecoord(mt,treecoord);
	TREECOORD* cur_path = treecoord;
	MOVELIST* ml = get_move_list(&pmove->chess);
	WIN_RATE* winrates_leaf_parent = getwinratebytreecoord(win_rates, treecoord);
	WIN_RATE* winrates_leaf = (WIN_RATE*)malloc(ml->count*sizeof(WIN_RATE));
	memset(winrates_leaf, 0, sizeof(WIN_RATE)*ml->count);
	winrates_leaf_parent->children = winrates_leaf;
	winrates_leaf->parent = winrates_leaf_parent;
	pmove->next = ml;
	int i=0;
	for (;i<ml->count;i++)
	{
		(winrates_leaf+i)->parent = winrates_leaf_parent;
		int winres = simulate(&(ml->move_list+i)->chess);
		(winrates_leaf + i)->count++;
		if ((winres == 1 && (ml->move_list+i)->chess.turn == RED) || (winres == 2 && (ml->move_list+i)->chess.turn == BLACK))
		{
			(winrates_leaf + i)->rwin+=1;
		}
		backward_m(win_rates, treecoord ,winres, (ml->move_list+i)->chess.turn);
	}

	return 0;
}

TREECOORD* selection(WIN_RATE* wr, MOVETREE* mt)
{
	int i = 0;
	int r = 0;
	double max = 0.0;
	CHESS* pcur_chess = NULL;
	WIN_RATE* pcur_winrate = wr;
	MOVELIST* ml = mt->root;
	TREECOORD* treecoord = init_treecoord(0);
	while(pcur_winrate)
	{
		r = 0;
		i = 0;
		for(;i<ml->count;i++)
		{
			double val = calc_ucb_value(pcur_winrate+i);
			if(val > max)
			{
				r = i;
				max = val;
			}
		}
		//printf("MAX VALUE %d %d %lf\n",treecoord->depth, r, max);
		pcur_winrate = (pcur_winrate+r)->children;
		ml = (ml->move_list+r)->next;
		TREECOORD* tmptreecoord = init_treecoord(treecoord->depth+1);
		if(treecoord->depth > 0)
			memcpy(tmptreecoord->index,treecoord->index,treecoord->depth*sizeof(INDEX));
		tmptreecoord->index[treecoord->depth].index=r;
		cleantreecoord(treecoord);
		treecoord =tmptreecoord;
	}
	return treecoord;
}

CHESS* monte_carlo_search(CHESS* pchess)
{
	int total = 0;
	MOVELIST* ml = get_move_list(pchess);
	MOVETREE* mt = (MOVETREE*)malloc(sizeof(MOVETREE));
	mt->root = ml;
	WIN_RATE* win_rates = (WIN_RATE*)malloc(sizeof(WIN_RATE)*ml->count);

	WIN_RATE* win_rates_parent = (WIN_RATE*)malloc(sizeof(WIN_RATE));
	memset(win_rates, 0, sizeof(WIN_RATE)*ml->count);
	memset(win_rates_parent, 0, sizeof(WIN_RATE));
	int i = 0;
	int depth = 1;
	TREECOORD* cur_path = init_treecoord(depth);
	MOVELIST* tmpl = ml;
	for(;i<ml->count;i++)
	{
		(win_rates+i)->parent = win_rates_parent;
		int rwin = simulate(&(ml->move_list+i)->chess);
		if((rwin == 1 && (ml->move_list+i)->chess.turn == RED) || (rwin == 2 && (ml->move_list+i)->chess.turn == BLACK))
		{
			win_rates_parent->rwin+=1;
			(win_rates+i)->rwin+=1;
		}
		(win_rates+i)->count+=1;
		(win_rates_parent)->count+=1;
	}
	i=0;
	while (total < 500)
	{
		total++;
		WIN_RATE* cur_win_rates = win_rates;
		TREECOORD* treecoord = selection(win_rates, mt);
		i=0;
		/*for(;i<treecoord->depth;i++)
		{
			printf(" %d", treecoord->index[i].index);
		}
		printf("\n");*/
		WIN_RATE* mwr = win_rates+treecoord->index[0].index;
		expansion(win_rates, mt, treecoord);
		cleantreecoord(treecoord);
	}
	double max = 0;
	int r=0;
	for(i=0;i<ml->count;i++)
	{
		double val = (double)(win_rates+i)->rwin/(win_rates+i)->count;
		//printf("%d %d %lf %lf\n",(win_rates+i)->count, i, val, calc_ucb_value(win_rates+i));
		if(val>max)
		{
			max=val;
			r=i;
		}
	}
	printf("%d %d %lf\n", pchess->turn, r, (double)(win_rates+r)->rwin/(win_rates+r)->count);
	printchessui(&(ml->move_list+r)->chess);
	free(win_rates);
	return &(ml->move_list+r)->chess;
}
typedef struct __ENTIRE_MATCH
{
	int count;
	double* data;
}ENTIRE_MATCH;
ENTIRE_MATCH* alloc_em()
{
	ENTIRE_MATCH* em =  (ENTIRE_MATCH*)malloc(sizeof(ENTIRE_MATCH));
	em->count = 0;
	em->data = NULL;
	return em;
}
void free_em(ENTIRE_MATCH* em)
{
	free(em->data);
	free(em);
}
void append(ENTIRE_MATCH* em, double* buffer)
{
	double* data = (double*)malloc(sizeof(double)*(em->count+1)*(BOARD_SIZE+1));
	if(em->count)
	{
		memcpy(data, em->data, sizeof(double)*(em->count)*(BOARD_SIZE+1));
		free(em->data);
	}
	memcpy(data+(em->count)*(BOARD_SIZE+1), buffer, (BOARD_SIZE+1)*sizeof(double));
	em->data = data;
	em->count+=1;
}
void update_result(ENTIRE_MATCH* em, double result)
{
	int i = 0;
	for(;i<em->count;i++)
	{
		int offset = i*(BOARD_SIZE+1)+BOARD_SIZE;
		em->data[offset] = result;
	}
}
#define TRUE 1L
#define FALSE 0L
int compare_mem(void* start, void* end, int len)
{
	char* pstart = (char*)start;
	char* pend = (char*)end;
	int i = 0;
	int flag = TRUE;
	for(;i<len;i++)
	{
		if(pstart[i] != pend[i])
		{
			flag = FALSE;
			return flag;
		}
	}
	return flag;
}
int getmaxvalue(NeuralNetWork* pNN, MOVELIST* ml, ENTIRE_MATCH* em)
{
	int sr = rand()%1;
	if(sr == 0)
		return rand()%ml->count;
	int i = 0;
	int r = 0;
	double max = 0.0;
	double* result = (double*)malloc(2*sizeof(double)*(pNN->hiddenWidth*pNN->hiddenCount+pNN->hiddenWidth*pNN->outWidth));
	for(;i<ml->count;i++)
	{
		CHESS* pchess = &(ml->move_list[i].chess);
		double* pbuffer = chess_to_buffer(pchess);
		int j = 0;
		int flag = FALSE;
		for(;j< em->count;j++)
		{
			if(compare_mem(em->data+j*92, pbuffer,91*sizeof(double)))
			{
				flag = TRUE;
				break;
			}
		}
		if(flag)
		{
			free(pbuffer);
			continue;
		}

		forward_g(pNN->data,pbuffer, result, pNN->inputWidth, pNN->hiddenWidth, pNN->hiddenCount, pNN->outWidth);

		double Ok = *(result+2*(pNN->hiddenWidth*pNN->hiddenCount));
		if(i == 0)
			max = Ok;


		if(pchess->turn == BLACK)
		{
			if(Ok>max)
			{
				max = Ok;
				r = i;
			}
		}
		else
		{
			if(Ok<max)
			{
				max = Ok;
				r = i;
			}
		}

		free(pbuffer);
	}
	free(result);
	return r;
}
int v_main()
{
	init_weight(gweight, BOARD_SIZE);
	CHESS* pchessbk = get_chess_from_fen("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr 1");
	//get_random_vs(pchessbk);
	//return 0;
	/*int idot = 0;
	for(;idot < 10000; idot++)
	{
		dot_chain(pchessbk);
	}
	return 0;*/
	//while (1)
	{
		double buffer_in[BOARD_SIZE] = { 0.0 };
		//fread(buffer_in, (BOARD_SIZE)*sizeof(double), 1, stdin);
		//CHESS* pchess = buffer_to_chess(buffer_in);
		CHESS* pchess = pchessbk;
		/*while(!check_end(pchess))
		{
			pchess = monte_carlo_search(pchess);
		}*/
		int log[BOARD_SIZE + 1] = { 0 };
		int flag[BOARD_SIZE + 1] = { 0 };
		int width = BOARD_SIZE + 1;
		int height = 0;
		double* matrix[1000] = { 0 };
		int i = 0;
		FILE* fp = fopen("cr.qp", "wb");
		MOVELIST* prev = NULL;
		ENTIRE_MATCH* em = alloc_em();
		NeuralNetWork* pNN = loadNeuralNetwork("nn.txt");
		if(!pNN)
			pNN = initNeuralNetwork(91,100,5,1);
		while(i<100)
		{
			//printf("count %d\n", i);
			//double* buffer = get_random_vs(pchess);
			CHESS* bchess = pchess;
			MOVELIST* plist = get_move_list(pchess);
			//int r = rand() % plist->count;
			int r = getmaxvalue(pNN, plist,em);
			int count = plist->count;
			int bestmove = 0;
			MOVELIST* ml = plist;
			float max = 0.0;
			int offset = 0;
			/*if (pchess->turn == BLACK)
			  {
			  for (; bestmove < count; bestmove++) {
			  float cur_weight = calc_weight(gweight, &((ml->move_list + bestmove)->chess));
			  if (bestmove == 0)
			  max = cur_weight;
			  if (pchess->turn == BLACK)
			  cur_weight *= -1;
			  if (cur_weight >= max) {
			  max = cur_weight;
			  offset = bestmove;
			  }
			  }
			  r = offset;
			  }*/
			CHESS* pchess_cur = &(plist->move_list[r].chess);
			pchess = pchess_cur;
			printchessui(pchess);
			printf("\n\n\n\n %d \n\n\n\n\n\n", em->count);

			double* buffer;
			buffer = chess_to_buffer(pchess);
			append(em,buffer);
			int ret = check_end(pchess_cur);
			if (ret)
			{
				//buffer = chess_to_buffer(pchess);
				if(ret == 1)
					buffer[BOARD_SIZE] = 1.0;
				else
					buffer[BOARD_SIZE] = 0.0;
				update_result(em, buffer[BOARD_SIZE]);
				fwrite(em->data, 92*sizeof(double), em->count, fp);
				//fwrite(buffer, 92 * sizeof(double), 1, fp);
				fflush(fp);
				printchessui(pchess);
				printf("\n\n");
				
				free_em(em);
				em = alloc_em();
				free(buffer);
				pchess = pchessbk;
				i++;
			}
			else
				//printf("Trainning Here\n");
				;//buffer = chess_to_buffer(pchess_cur);
			//addtomatrix(matrix, width, &height, log, flag, buffer);
			//printf("round %d\n", i);
			//free(buffer);
			//free(pchess);
			if(prev)
			{
				free(prev->move_list);
				free(prev);
			}
			prev = plist;
			/*free(plist->move_list);
			free(plist);*/
		}
		fclose(fp);

		return 0;
		//fclose(fp);
		for (i = 0; i<height; i++)
		{
			int j = 0;
			for (; j<width; j++)
			{
				//printf("%lf ",matrix[log[i]][j]);
			}
			//printf("\n");
		}
	}

}
#ifdef _NM_
int main(int argc, char** argv)
{
	srand(time(NULL));
	return v_main();
}
#endif
