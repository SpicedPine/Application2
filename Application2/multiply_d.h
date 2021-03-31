#include "omp.h"

#define NUM 1025
void multiply_d(double a[][NUM], double b[][NUM], double c[][NUM])
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < NUM; i++) {
		for (j = 0; j < NUM; j++) {
			//2 #pragma omp parallel for
			for (k = 0; k < NUM; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}

void sum_d(double a[][NUM], double b[][NUM], double c[][NUM]) {
	int i, j;
#pragma omp parallel for
	for (i = 0; i < NUM; i++)
	{
		for (j = 0; j < NUM; j++)
		{
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}
