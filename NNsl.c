#include "NNsl.h"
#include "math.h"
#include "stdio.h"
void forward_g(double* net,double* data,double* result, int inputWidth, int hiddenWidth, int hiddenCount, int outWidth)
{
	int i = 0;
	for(;i<hiddenWidth;i++)
	{
		int j = 0;
		*(result+2*i) = 0;
		for(;j<inputWidth;j++)
		{
			*(result+2*i) += *(net+i*inputWidth+j)*(*(data+j));
		}
		*(result+2*i) = 1.0/(1.0+exp(-(*(result+2*i))));
	}

	int index = 1;
	for(;index<hiddenCount;index++)
	{
		for(i=0;i<hiddenWidth;i++)
		{
			*(result+2*index*hiddenWidth+2*i)=0;
			int j = 0;
			for(;j<hiddenWidth;j++)
			{
				*(result+2*(index*hiddenWidth+i)) += *(net+(inputWidth*hiddenWidth)+(index-1)*hiddenWidth*hiddenWidth+hiddenWidth*i+j)*(*(result+2*((index-1)*hiddenWidth+j)));
			}
			*(result+2*index*hiddenWidth+2*i) += *(net+(inputWidth*hiddenWidth+(hiddenWidth*hiddenWidth)*(hiddenCount-1)+hiddenWidth*outWidth)    + (index-1)*hiddenWidth+i);
			*(result+2*index*hiddenWidth+2*i) = 1.0/(1.0+exp(-(*(result+2*index*hiddenWidth+2*i))));
		}
	}
	i = 0;
	for(;i<outWidth;i++)
	{
		int j = 0;
		*(result+2*i+2*hiddenCount*hiddenWidth) = 0;
		for(;j<hiddenWidth;j++)
		{
			*(result+2*i+2*hiddenCount*hiddenWidth) += *(net+hiddenWidth*inputWidth+(hiddenWidth*hiddenWidth*(hiddenCount-1))+i*hiddenWidth+j)*(*(result+2*(hiddenCount-1)*hiddenWidth+j*2));
		}
		*(result+2*i+2*hiddenCount*hiddenWidth) += *(net+(inputWidth*hiddenWidth+(hiddenWidth*hiddenWidth)*(hiddenCount-1)+hiddenWidth*outWidth)    + (hiddenCount-1)*hiddenWidth+i);
		*(result+2*i+2*hiddenCount*hiddenWidth) = 1.0/(1.0+exp(-(*(result+2*i+2*hiddenCount*hiddenWidth))));
	}

}
void backward_g(double* net,double* data,double* result, int inputWidth, int hiddenWidth, int hiddenCount, int outWidth)
{
	int i = 0;
	for(;i<outWidth;i++)
	{
		double Ok = *(result+2*(hiddenWidth*hiddenCount+i));
		double Tk = *(data+inputWidth+i);
		*(result+2*(hiddenWidth*hiddenCount+i)+1) = (1-Ok)*Ok*(Ok-Tk);
	}
	i = hiddenCount-1;
	for(;i>=hiddenCount-1;i--)
	{
		int j = 0;
		for(;j<hiddenWidth;j++)
		{
			double delta = 0.0;
			*(result+2*(hiddenWidth*i+j)+1) = 0;
			int k = 0;
			for(;k<outWidth;k++)
			{
				delta += *(result+2*(hiddenWidth*(i+1)+k)+1)*(*(net+hiddenWidth*inputWidth+(i)*hiddenWidth*hiddenWidth+k*hiddenWidth+j));
			}
			*(result+2*(hiddenWidth*i+j)+1) = delta*(1.0-(*(result+2*(hiddenWidth*(i)+j))))*(*(result+2*(hiddenWidth*(i)+j)));
		}

	}
	for(;i>=0;i--)
	{
		int j = 0;
		for(;j<hiddenWidth;j++)
		{
			double delta = 0.0;
			*(result+2*(hiddenWidth*i+j)+1) = 0;
			int k = 0;
			for(;k<hiddenWidth;k++)
			{
				delta += *(result+2*(hiddenWidth*(i+1)+k)+1)*(*(net+hiddenWidth*inputWidth+(i)*hiddenWidth*hiddenWidth+k*hiddenWidth+j));
			}
			*(result+2*(hiddenWidth*i+j)+1) = delta*(1.0-(*(result+2*(hiddenWidth*(i)+j))))*(*(result+2*(hiddenWidth*(i)+j)));
		}
	}

	for(i=hiddenCount;i>=hiddenCount;i--)
	{
		int j=0;
		for(;j<outWidth;j++)
		{
			int left = 0;
			for(;left<hiddenWidth;left++)
			{
				*(net+hiddenWidth*inputWidth+(hiddenWidth*hiddenWidth)*(hiddenCount-1)+j*hiddenWidth+left) += -0.005*(*(result+2*(hiddenWidth*hiddenCount+j)+1))*(*(result+2*(hiddenWidth*(hiddenCount-1)+left)));
			}
			*(net+inputWidth*hiddenWidth+(hiddenCount-1)*hiddenWidth*hiddenWidth+outWidth*hiddenWidth+(hiddenCount-1)*hiddenWidth+j) += -0.005*(*(result+2*(hiddenWidth*hiddenCount+j)+1));
		}
	}

	i = hiddenCount-1;

	for(;i>0;i--)
	{
		int j=0;
		for(;j<hiddenWidth;j++)
		{
			int left = 0;
			for(;left<hiddenWidth;left++)
			{
				*(net+hiddenWidth*inputWidth+(hiddenWidth*hiddenWidth)*(i-1)+j*hiddenWidth+left) += -0.005*(*(result+2*(hiddenWidth*i+j)+1)*(*(result+2*(hiddenWidth*(i-1)+left))));
			}
			*(net+inputWidth*hiddenWidth+(hiddenCount-1)*hiddenWidth*hiddenWidth+outWidth*hiddenWidth+(i-1)*hiddenWidth+j) += -0.005*(*(result+2*(hiddenWidth*i+j)+1));
		}
	}

	int j=0;
	for(;j<hiddenWidth;j++)
	{
		int left = 0;
		for(;left<inputWidth;left++)
		{
			*(net+j*inputWidth+left) += -0.005*(*(result+2*(j)+1)*(*(data+left)));
		}
	}
}
void train_g(double* net, double* data, double* result, int inputWidth, int hiddenWidth, int hiddenCount, int outWidth, int recCount)
{
	int epoch = 0;
	for(;epoch<1;epoch++)
	{
		int i = 0;
		for(;i<recCount;i++)
		{
			forward_g(net, data+i*(inputWidth+outWidth), result, inputWidth, hiddenWidth, hiddenCount, outWidth);
			//printf("last %.16lf %lf\n", result[hiddenWidth*hiddenCount*2], *(data+i*92+91));
			backward_g(net, data+i*(inputWidth+outWidth), result, inputWidth, hiddenWidth, hiddenCount, outWidth);
		}
	}
}
