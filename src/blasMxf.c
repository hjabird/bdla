#include "libbdla.h"
/*============================================================================
blasVxf.c

Floating point matrix basic linear algebra.

Copyright(c) 2019 HJA Bird

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
============================================================================*/
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <openblas/cblas.h>

BDLA_EXPORT bdla_Mxf bdla_Mxf_create(int r, int c) {
	assert(r > 0);
	assert(c > 0);
	bdla_Mxf ret = { r, c, malloc(sizeof(float)*c*r) };
	return ret;
}

BDLA_EXPORT void bdla_Mxf_release(bdla_Mxf *mat) {
	if (mat != NULL) {
		assert(mat->arr != NULL);
		free(mat->arr); mat->arr = NULL;
		mat->rows = 0;
		mat->cols = 0;
	}
	return;
}

BDLA_EXPORT bdla_Mxf bdla_Mxf_copy(bdla_Mxf mat) {
	assert(mat.arr != NULL);
	bdla_Mxf ret = mat;
	ret.arr = malloc(sizeof(float) * ret.cols * ret.rows);
	memcpy(ret.arr, mat.arr, sizeof(float) * ret.cols * ret.rows);
	return ret;
}

BDLA_EXPORT int bdla_Mxf_rows(bdla_Mxf A) {
	assert(A.arr != NULL);
	return A.rows;
}

BDLA_EXPORT int bdla_Mxf_cols(bdla_Mxf A) {
	assert(A.arr != NULL);
	return A.cols;
}

BDLA_EXPORT int bdla_Mxf_isequal(bdla_Mxf A, bdla_Mxf B) {
	assert(A.arr != NULL);
	assert(B.arr != NULL);
	assert(A.cols >= 0);
	assert(B.cols >= 0);
	assert(A.rows >= 0);
	assert(B.rows >= 0);
	if (A.rows != B.rows) { return 0; }
	else if (A.cols != B.cols) { return 0; }
	else {
		if (!memcmp(A.arr, B.arr, sizeof(float) * A.cols * A.rows)) { 
			return 1; 
		}
		else { return 0; }
	}
}

BDLA_EXPORT bdla_Status bdla_Mxf_zero(bdla_Mxf *A) {
	assert(A != NULL);
	assert(A->arr != NULL);
	memset(A->arr, 0x0, sizeof(float)*A->cols*A->rows);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_plus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->cols > 0);
	assert(Y->rows > 0);
	assert(A.arr != NULL);
	assert(A.cols > 0);
	assert(A.rows > 0);
	assert(B.arr != NULL);
	assert(B.cols > 0);
	assert(B.rows > 0);

	int i, max;
	if (A.cols != B.cols || A.rows != B.rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	max = A.cols * A.rows;
	/* Should work fine inplace. */
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] + B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_fplus(bdla_Mxf A, float b, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] + b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_minus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->cols > 0);
	assert(Y->rows > 0);
	assert(A.arr != NULL);
	assert(A.cols > 0);
	assert(A.rows > 0);
	assert(B.arr != NULL);
	assert(B.cols > 0);
	assert(B.rows > 0);

	int i, max;
	if (A.cols != B.cols && A.rows != B.rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.cols != Y->cols && A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	max = A.cols * A.rows;
	/* Should work fine inplace. */
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] - B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_fminus(bdla_Mxf A, float b, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] - b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_fmult(bdla_Mxf A, float b, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != Y->cols && A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] * b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_ewmult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != B.cols || A.rows != B.rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] * B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_mult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != B.rows) { return BDLA_DIMENSION_MISMATCH; }
	if (A.rows != Y->cols || A.cols != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int alias = 0;
	float *outarr = Y->arr;
	if (Y->arr == A.arr) {
		alias = 1;
		outarr = malloc(sizeof(float) * Y->cols * Y->rows);
		if (outarr == NULL) { return BDLA_MEM_ERROR; }
	}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows,
		B.cols, B.rows, 1.f, A.arr, A.cols, B.arr, B.cols, 0.f, outarr, Y->cols);
	if (alias) {
		Y->arr = outarr;
		free(A.arr);
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_vmult(bdla_Mxf A, bdla_Vxf b, bdla_Vxf *y) {
	assert(A.arr != NULL);
	assert(A.cols >= 0);
	assert(A.rows >= 0);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	assert(b.len >= 0);
	assert(b.arr != NULL);
	if (A.rows != y->len || A.cols != b.len) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int alias = 0;
	float *tmparr = y->arr;
	if (y->arr == b.arr) {
		alias = 1;
		tmparr = malloc(sizeof(float) * y->len);
	}
	cblas_sgemv(CblasRowMajor, CblasNoTrans, A.rows, A.cols, 1.f,
		A.arr, A.cols, b.arr, 1, 0.f, y->arr, 1);
	if(alias){
		free(y->arr);
		y->arr = tmparr;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_fdiv(bdla_Mxf A, float b, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] / b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_ewdiv(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.cols != B.cols || A.rows != B.rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.cols != Y->cols || A.rows != Y->rows) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int i, max;
	max = A.cols * A.rows;
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] / B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_row(bdla_Mxf A, int row, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	if (A.cols != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (row < 0 || row > A.rows) { return BDLA_BAD_INDEX; }
	memcpy(y->arr, &A.arr[row * A.cols], sizeof(float)*A.cols);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writerow(bdla_Mxf A, int row, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	if (A.cols != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (row < 0 || row > A.rows) { return BDLA_BAD_INDEX; }
	memcpy(&A.arr[row * A.cols], y->arr, sizeof(float)*A.cols);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_col(bdla_Mxf A, int col, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	if (A.rows != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (col < 0 || col > A.cols) { return BDLA_BAD_INDEX; }
	int i;
	for (i = 0; i < A.rows; ++i) {
		y->arr[i] = A.arr[col + i * A.cols];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writecol(bdla_Mxf A, int col, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	if (A.rows != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (col < 0 || col > A.cols) { return BDLA_BAD_INDEX; }
	int i;
	for (i = 0; i < A.rows; ++i) {
		A.arr[col + i * A.cols] = y->arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_submat(bdla_Mxf A, int row, int col, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr == NULL);
	assert(Y->cols >= 0);
	assert(Y->rows >= 0);
	assert(A.arr != NULL);
	assert(A.rows >= 0);
	assert(A.cols >= 0);
	if (row < 0 || col < 0) { return BDLA_BAD_INDEX; }
	if (A.rows < row + Y->rows) { return BDLA_BAD_INDEX; }
	if (A.cols < col + Y->cols) { return BDLA_BAD_INDEX; }
	int i, j, k, maxi, maxj;
	maxi = row + Y->rows;
	maxj = col + Y->cols;
	k = 0;
	for (i = row; i < maxi; ++i) {
		for (j = col; j < maxj; ++j) {
			Y->arr[k] = A.arr[i * A.cols + j];
			++k;
		}
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writesubmat(bdla_Mxf A, int row, int col, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr == NULL);
	assert(Y->cols >= 0);
	assert(Y->rows >= 0);
	assert(A.arr != NULL);
	assert(A.rows >= 0);
	assert(A.cols >= 0);
	if (row < 0 || col < 0) { return BDLA_BAD_INDEX; }
	if (A.rows < row + Y->rows) { return BDLA_BAD_INDEX; }
	if (A.cols < col + Y->cols) { return BDLA_BAD_INDEX; }
	int i, j, k, maxi, maxj;
	maxi = row + Y->rows;
	maxj = col + Y->cols;
	k = 0;
	for (i = row; i < maxi; ++i) {
		for (j = col; j < maxj; ++j) {
			A.arr[i * A.cols + j] = Y->arr[k];
			++k;
		}
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_uniform(bdla_Mxf *A, float b) {
	assert(A->arr != NULL);
	assert(A->rows >= 0);
	assert(A->cols >= 0);
	int i, maxi;
	maxi = A->rows * A->cols;
	for (i = 0; i < maxi; ++i) {
		A->arr[i] = b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_eye(bdla_Mxf *A) {
	assert(A->arr != NULL);
	assert(A->rows >= 0);
	assert(A->cols >= 0);
	if (A->rows != A->cols) { return BDLA_NONSQUARE; }
	int i, maxi;
	maxi = A->rows * A->rows;
	bdla_Status stat = BDLA_GOOD;
	stat = bdla_Mxf_zero(A);
	if (stat == BDLA_GOOD) {
		for (i = 0; i < maxi; i += A->rows + 1 ) {
			A->arr[i] = 1.f;
		}
		return BDLA_GOOD;
	}
	else
	{
		return stat;
	}
}

BDLA_EXPORT bdla_Status bdla_Mxf_diag(bdla_Mxf *A, bdla_Vxf b, int k) {
	assert(A->arr != NULL);
	assert(A->rows >= 0);
	assert(A->cols >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	if (A->rows != A->cols) { return BDLA_NONSQUARE; }
	if (A->rows - abs(k) != b.len) { return BDLA_DIMENSION_MISMATCH; }
	int i, j, maxi;
	bdla_Status stat = BDLA_GOOD;
	i = k >= 0 ? k : abs(k)*A->cols;
	maxi = k >= 0 ? A->rows * (A->rows - abs(k)) : A->rows * A->rows;
	stat = bdla_Mxf_zero(A);
	if (stat == BDLA_GOOD) {
		for (j = 0; i < maxi; i += A->rows + 1, ++j) {
			A->arr[i] = b.arr[j];
		}
	}
	return stat;
}
