
/*------------------------------------------------------------------------------*/
// matrix_mult.cc -- Implementation of matrix multiplication with
// Strassen's algorithm.
//
// Compile this file with gcc command:
// g++ -Wall -o matrix_mult matrix_mult.cc 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#define aMHZ 667
using namespace std;

struct timespec timer[6];
double tdiff[3];

// This function allocates the matrix
inline double** allocate_matrix(int n)
{
	double** mat = new double*[n];
	for (int i = 0;i<n;++i)
	{
		mat[i] = new double[n];
		memset(mat[i], 0, sizeof(double)*n);
	}

	return (mat); // returns the pointer to the vector.
}

/*------------------------------------------------------------------------------*/
// This function unallocates the matrix (frees memory)
inline void free_matrix(double **M, int n)
{
	for (int i = 0; i < n; i++)
	{
		delete[] M[i];
	}

	delete[] M; // frees the pointer /
	M = NULL;
}

/*------------------------------------------------------------------------------*/
// function to sum two matrices
inline void sum(double **a, double **b, double **result, int tam) {

	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			result[i][j] = a[i][j] + b[i][j];
		}
	}
}

/*------------------------------------------------------------------------------*/
// function to subtract two matrices
inline void subtract(double **a, double **b, double **result, int tam) {

	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			result[i][j] = a[i][j] - b[i][j];
		}
	}
}

/*------------------------------------------------------------------------------*/
// naive method
void naive(double** A, double** B, double** C, int n)
{
	for (int i = 0;i<n;i++)
		for (int j = 0;j<n;j++)
			for (int k = 0;k<n;k++)
				C[i][j] += A[i][k] * B[k][j];
}

/*------------------------------------------------------------------------------*/
// Strassen's method
void strassen(double **a, double **b, double **c, int tam)
{
	// Key observation: call naive method for matrices smaller than 2 x 2
	if (tam <= 4)
	{
		naive(a, b, c, tam);
		return;
	}

	// other cases are treated here:
	int newTam = tam / 2;
	double **a11, **a12, **a21, **a22;
	double **b11, **b12, **b21, **b22;
	double **c11, **c12, **c21, **c22;
	double **p1, **p2, **p3, **p4, **p5, **p6, **p7;

	// memory allocation:
	a11 = allocate_matrix(newTam);
	a12 = allocate_matrix(newTam);
	a21 = allocate_matrix(newTam);
	a22 = allocate_matrix(newTam);

	b11 = allocate_matrix(newTam);
	b12 = allocate_matrix(newTam);
	b21 = allocate_matrix(newTam);
	b22 = allocate_matrix(newTam);

	c11 = allocate_matrix(newTam);
	c12 = allocate_matrix(newTam);
	c21 = allocate_matrix(newTam);
	c22 = allocate_matrix(newTam);

	p1 = allocate_matrix(newTam);
	p2 = allocate_matrix(newTam);
	p3 = allocate_matrix(newTam);
	p4 = allocate_matrix(newTam);
	p5 = allocate_matrix(newTam);
	p6 = allocate_matrix(newTam);
	p7 = allocate_matrix(newTam);

	double **aResult = allocate_matrix(newTam);
	double **bResult = allocate_matrix(newTam);

	//dividing the matrices in 4 sub-matrices:
	for (int i = 0; i < newTam; i++) {
		for (int j = 0; j < newTam; j++) {
			a11[i][j] = a[i][j];
			a12[i][j] = a[i][j + newTam];
			a21[i][j] = a[i + newTam][j];
			a22[i][j] = a[i + newTam][j + newTam];

			b11[i][j] = b[i][j];
			b12[i][j] = b[i][j + newTam];
			b21[i][j] = b[i + newTam][j];
			b22[i][j] = b[i + newTam][j + newTam];
		}
	}

	// Calculating p1 to p7:

	sum(a11, a22, aResult, newTam); // a11 + a22
	sum(b11, b22, bResult, newTam); // b11 + b22
	strassen(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

	sum(a21, a22, aResult, newTam); // a21 + a22
	strassen(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

	subtract(b12, b22, bResult, newTam); // b12 - b22
	strassen(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

	subtract(b21, b11, bResult, newTam); // b21 - b11
	strassen(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

	sum(a11, a12, aResult, newTam); // a11 + a12
	strassen(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

	subtract(a21, a11, aResult, newTam); // a21 - a11
	sum(b11, b12, bResult, newTam); // b11 + b12
	strassen(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

	subtract(a12, a22, aResult, newTam); // a12 - a22
	sum(b21, b22, bResult, newTam); // b21 + b22
	strassen(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

											// calculating c21, c21, c11 e c22:

	sum(p3, p5, c12, newTam); // c12 = p3 + p5
	sum(p2, p4, c21, newTam); // c21 = p2 + p4

	sum(p1, p4, aResult, newTam); // p1 + p4
	sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
	subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

	sum(p1, p3, aResult, newTam); // p1 + p3
	sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
	subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

	// Grouping the results obtained in a single matrix:
	for (int i = 0; i < newTam; i++) {
		for (int j = 0; j < newTam; j++) {
			c[i][j] = c11[i][j];
			c[i][j + newTam] = c12[i][j];
			c[i + newTam][j] = c21[i][j];
			c[i + newTam][j + newTam] = c22[i][j];
		}
	}

	// deallocating memory (free):
	free_matrix(a11, newTam);
	free_matrix(a12, newTam);
	free_matrix(a21, newTam);
	free_matrix(a22, newTam);

	free_matrix(b11, newTam);
	free_matrix(b12, newTam);
	free_matrix(b21, newTam);
	free_matrix(b22, newTam);

	free_matrix(c11, newTam);
	free_matrix(c12, newTam);
	free_matrix(c21, newTam);
	free_matrix(c22, newTam);

	free_matrix(p1, newTam);
	free_matrix(p2, newTam);
	free_matrix(p3, newTam);
	free_matrix(p4, newTam);
	free_matrix(p5, newTam);
	free_matrix(p6, newTam);
	free_matrix(p7, newTam);
	free_matrix(aResult, newTam);
	free_matrix(bResult, newTam);

} // end of Strassen function

  /*------------------------------------------------------------------------------*/
  // Generate random matrices
void gen_matrix(double** M, int n)
{
	for (int i = 0;i<n;++i)
	{
		for (int j = 0;j<n;++j)
		{
			M[i][j] = rand() % 11;
			//M[i][j]=1;
		}
	}
}


void print_matrix(double** M, int n)
{
	for (int i = 0;i<n;++i)
	{
		for (int j = 0;j<n;++j)
		{
			printf("%lf", M[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}



int main(int argc, char** argv)
{
	srand(time(NULL));

	int mdim = 512; // matrix dimension
	bool is_strassen = true;
	

	// create new matrices
	double** A = allocate_matrix(mdim);
	double** B = allocate_matrix(mdim);
	double** C = allocate_matrix(mdim);
	gen_matrix(A, mdim);
	gen_matrix(B, mdim);

	// matrices multiplication
	if (is_strassen) {
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &timer[2]);
		strassen(A, B, C, mdim);
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &timer[3]);

	}
	else
		naive(A, B, C, mdim);
	
	//Time difference
	tdiff[1] = (timer[3].tv_sec - timer[2].tv_sec) * 1000 + ((double)(timer[3].tv_nsec - timer[2].tv_nsec) / 1000000.0);
	
	//matrix print
	//print_matrix(C, mdim);
	printf("---------------------------------------------\n");
	printf("Calculating:   C[512][512] = A[512][512] * B[512][512]\n");
	printf("Host(time)          %9.1fmsec (@ %03d MHZ)\n", tdiff[1], aMHZ);
	printf("---------------------------------------------\n");
	free_matrix(A, mdim);
	free_matrix(B, mdim);
	free_matrix(C, mdim);

	return 0;
}