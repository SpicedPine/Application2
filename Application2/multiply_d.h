#include "omp.h"
#include "mkl.h"

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

void multiply_d_mkl(double a[][NUM], double b[][NUM], double c[][NUM]) {
	int i, j;
	for ( i = 0; i < NUM; i++)
	{
#pragma omp parallel for
		for (j = 0; j < NUM; j++)
		{
			c[j][i] = cblas_ddot(NUM, a[j], 1, b[NUM - i - 1], 1);
		}
	}

}

void rotate(double a[][NUM], double b[][NUM]) {
#pragma omp parallel for
	for (size_t i = 0; i < NUM; i++)
	{
		for (size_t j = 0; j < NUM; j++)
		{
			b[NUM - j - 1][i] = a[i][j];
		}
	}
}
