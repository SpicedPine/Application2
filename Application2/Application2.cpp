//  Simple minded matrix multiply
#include <stdio.h>
#include "multiply_d.h"

//#include <emmintrin.h>

static double  a[NUM][NUM], b[NUM][NUM], c[NUM][NUM];

//routine to initialize an array with data
void init_arr(double row, double col, double off, double a[][NUM])
{
	int i, j;

	for (i=0; i< NUM;i++) {
		for (j=0; j<NUM;j++) {
			a[i][j] = row*i+col*j+off;
		}
	}
}



// routine to print out contents of small arrays
void print_arr(const char * name, double array[][NUM])
{
	int i,j;
	
	printf("\n%s\n", name);
	for (i=0;i<NUM;i++){
		for (j=0;j<NUM;j++) {
			printf("%g\t",array[i][j]);
		}
		printf("\n");
	}
}


int main()
{
	double start, stop;

// initialize the arrays with data
	init_arr(3,-2,1,a);
	init_arr(-2,1,3,b);

	//start timing the matrix multiply code
	printf("NUM:%d\n",NUM);
	start = omp_get_wtime();
	multiply_d(a,b,c);
	stop = omp_get_wtime();

// print simple test case of data to be sure multiplication is correct


	if (NUM < 5) {
		print_arr("a", a);
		print_arr("b", b);
		print_arr("c", c);
	}

// print elapsed time
	printf("Elapsed time = %lf seconds\n", ((double)(stop - start)) );

}