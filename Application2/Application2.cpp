#include <stdio.h>
#include "multiply_d.h"
#include <time.h> 

static double  a[NUM][NUM], b[NUM][NUM], c[NUM][NUM], a2[NUM][NUM], a2b[NUM][NUM], c2[NUM][NUM], res[NUM][NUM], a_rot[NUM][NUM], b_rot[NUM][NUM], c_rot[NUM][NUM];

//routine to initialize an array with data
void init_arr(double row, double col, double off, double a[][NUM])
{
	int i, j;

	for (i = 0; i < NUM; i++) {
		for (j = 0; j < NUM; j++) {
			a[i][j] = row * i + col * j + off;
		}
	}
}



// routine to print out contents of small arrays
void print_arr(const char* name, double array[][NUM])
{
	int i, j;

	printf("\n%s\n", name);
	for (i = 0; i < NUM; i++) {
		for (j = 0; j < NUM; j++) {
			printf("%g\t", array[i][j]);
		}
		printf("\n");
	}
}


int main()
{
	double start, stop;

	// initialize the arrays with data
	init_arr(3, -2, 1, a);
	init_arr(-2, 1, 3, b);
	init_arr(-5, -2, 2, c);

	//start timing the matrix multiply code
	printf("NUM:%d\n", NUM);
	/*for (size_t j = 13; j < 17; j++)
	{
		omp_set_num_threads(j);
		printf("Number of threads = %d \n\n", j);
		for (size_t i = 0; i < 5; i++)
		{
			printf("Without MKL:");
			start = clock();
			multiply_d(a, a, a2);
			multiply_d(a2, b, a2b);
			multiply_d(c, c, c2);
			sum_d(a2b, c2, res);
			stop = clock();
			printf("Elapsed time without mkl = %lf seconds\n", ((double)(stop - start)) / 1000);
			printf("With MKL:");
			start = clock();
			rotate(a, a_rot);
			multiply_d_mkl(a, a_rot, a2);
			rotate(b, b_rot);
			multiply_d_mkl(a2, b_rot, a2b);
			rotate(c, c_rot);
			multiply_d_mkl(c, c_rot, c2);
			sum_d(a2b, c2, res);
			stop = clock();
			printf("Elapsed time with mkl = %lf seconds\n", ((double)(stop - start)) / 1000);
		}
	}*/

	omp_set_num_threads(16);

	printf("With MKL:");
	start = clock();
	rotate(a, a_rot);
	multiply_d_mkl(a, a_rot, a2);
	rotate(b, b_rot);
	multiply_d_mkl(a2, b_rot, a2b);
	rotate(c, c_rot);
	multiply_d_mkl(c, c_rot, c2);
	sum_d(a2b, c2, res);
	stop = clock();
	printf("Elapsed time with mkl = %lf seconds\n", ((double)(stop - start)) / 1000);


	/*printf("Without MKL:");
	start = clock();
	multiply_d(a, a, a2);
	multiply_d(a2, b, a2b);
	multiply_d(c, c, c2);
	sum_d(a2b, c2, res);
	stop = clock();
	printf("Elapsed time without mkl = %lf seconds\n", ((double)(stop - start)) / 1000);*/


	/*for (size_t j = 1; j < 21; j++)
	{
		omp_set_num_threads(j);
		printf("number of threads:%d\n\n", j);
		for (size_t i = 0; i < 5; i++)
		{
			start = omp_get_wtime();
			multiply_d(a, a, a2);
			multiply_d(a2, b, a2b);
			multiply_d(c, c, c2);
			sum_d(a2b, c2, res);
			stop = omp_get_wtime();
			printf("Elapsed time = %lf seconds\n", ((double)(stop - start)));
		}
	}*/

	// print simple test case of data to be sure multiplication is correct


	if (NUM < 5) {
		print_arr("a", a);
		print_arr("a", a);
		print_arr("a^2", a2);
	}

	// print elapsed time
	//printf("Elapsed time = %lf seconds\n", ((double)(stop - start)));

}
