#include "omp.h"

#define NUM 300
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
