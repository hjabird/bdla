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
		mat->dims[0] = 0;
		mat->dims[1] = 0;
	}
	return;
}

BDLA_EXPORT bdla_Mxf bdla_Mxf_copy(bdla_Mxf mat) {
	assert(mat.arr != NULL);
	bdla_Mxf ret = mat;
	ret.arr = malloc(sizeof(float) * ret.dims[1] * ret.dims[0]);
	memcpy(ret.arr, mat.arr, sizeof(float) * ret.dims[1] * ret.dims[0]);
	return ret;
}

BDLA_EXPORT void bdla_Mxf_transpose(bdla_Mxf A, bdla_Mxf *Y) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[0] > 0);
	assert(Y->dims[1] > 0);
	int alias = 0, i, j, im, jm;
	if (Y->arr == A.arr) {
		alias = 1;
		Y->arr = malloc(sizeof(float) * A.dims[0] * A.dims[1]);
		Y->dims[0] = A.dims[1];
		Y->dims[1] = A.dims[0];
	}
	im = A.dims[0];
	jm = A.dims[1];
	for (i = 0; i < im; ++i) {
		for (j = 0; j < jm; ++j) {
			bdla_Mxf_writevalue(*Y, j, i, bdla_Mxf_value(A, i, j));
		}
	}
	if (alias) {
		free(A.arr);
	}
}

BDLA_EXPORT bdla_Status bdla_Mxf_reshape(bdla_Mxf A, int rows, int cols, bdla_Mxf *Y){
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[0] > 0);
	assert(Y->dims[1] > 0);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(rows > 0);
	assert(cols > 0);
	if (rows * cols != A.dims[0] * A.dims[1]) {
		return BDLA_DIMENSION_MISMATCH;
	}
	int alias = 0, i, im;
	if (Y->arr = A.arr) {
		alias = 1;
		Y->arr = malloc(sizeof(float) * rows * cols);
		if (Y->arr == NULL) { return BDLA_MEM_ERROR; }
	}
	Y->dims[0] = rows;
	Y->dims[1] = cols;
	im = rows * cols;
	for (i = 0; i < im; ++i) {
		bdla_Mxf_writevalue(*Y, i%rows, i / rows,
			bdla_Mxf_value(A, i%A.dims[0], i / A.dims[1]));
	}
	if (alias) {
		free(A.arr);
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_resize(bdla_Mxf *A, int rows, int cols) {
	assert(A != NULL);
	assert(A->arr != NULL);
	assert(A->dims[0] > 0);
	assert(A->dims[1] > 0);
	assert(rows > 0);
	assert(cols > 0);
	A->dims[0] = rows;
	A->dims[1] = cols;
	int size = rows * cols;
	A->dims[0] = rows;
	A->dims[1] = cols;
	if (A->dims[0] * A->dims[1] != size) {
		A->arr = realloc(A->arr, sizeof(float) * size);
	}
	if(A->arr == NULL){
		return BDLA_MEM_ERROR;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT int bdla_Mxf_rows(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	return A.dims[0];
}

BDLA_EXPORT int bdla_Mxf_cols(bdla_Mxf A) {
	assert(A.arr != NULL);
	return A.dims[1];
}

BDLA_EXPORT int bdla_Mxf_isequal(bdla_Mxf A, bdla_Mxf B) {
	assert(A.arr != NULL);
	assert(B.arr != NULL);
	assert(A.dims[1] > 0);
	assert(B.dims[1] > 0);
	assert(A.dims[0] > 0);
	assert(B.dims[0] > 0);
	if (A.dims[0] != B.dims[0]) { return 0; }
	else if (A.dims[1] != B.dims[1]) { return 0; }
	else {
		if (!memcmp(A.arr, B.arr, sizeof(float) * A.dims[1] * A.dims[0])) { 
			return 1; 
		}
		else { return 0; }
	}
}

BDLA_EXPORT int bdla_Mxf_issquare(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	return A.dims[0] == A.dims[1] ? 1 : 0;
}

BDLA_EXPORT int bdla_Mxf_issymmetric(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	int i, j, im, jm, is_sym = 1;
	if (A.dims[0] != A.dims[1]) { is_sym = 0; }
	im = A.dims[0];
	for (i = 0; i < im; ++i) {
		jm = A.dims[1];
		for (j = i; j < jm; ++j) {
			if (bdla_Mxf_value(A, i, j) != bdla_Mxf_value(A, j, i)) { 
				is_sym = 0;
				break; 
			}
		}
		if (is_sym == 0) { break; }
	}
	return is_sym;
}

BDLA_EXPORT int bdla_Mxf_isdiagonal(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	int i, j, im, jm, is_diag = 1;
	if (A.dims[0] != A.dims[1]) { is_diag = 0; }
	im = A.dims[0];
	jm = A.dims[1];
	for (i = 0; i < im; ++i) {
		for (j = 0; j < i; ++j) {
			if (bdla_Mxf_value(A, i, j) != 0) {
				is_diag = 0;
				break;
			}
		}
		if (is_diag == 0) { break; }
		for (j = i+1; j < jm; ++j) {
			if (bdla_Mxf_value(A, i, j) != 0) {
				is_diag = 0;
				break;
			}
		}
		if (is_diag == 0) { break; }
	}
	return is_diag;
}

BDLA_EXPORT int bdla_Mxf_istrilower(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	int i, j, im, jm, is_tril = 1;
	if (A.dims[0] != A.dims[1]) { is_tril = 0; }
	im = A.dims[0];
	jm = A.dims[1];
	for (i = 0; i < im - 1; ++i) {
		for (j = i + 1; j < jm; ++j) {
			if (bdla_Mxf_value(A, i, j) != 0) {
				is_tril = 0;
				break;
			}
		}
		if (is_tril == 0) { break; }
	}
	return is_tril;
}

BDLA_EXPORT int bdla_Mxf_istriupper(bdla_Mxf A) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	int i, j, im, jm, is_triu = 1;
	if (A.dims[0] != A.dims[1]) { is_triu = 0; }
	im = A.dims[0];
	jm = A.dims[1];
	for (i = 1; i < im; ++i) {
		for (j = 0; j < i; ++j) {
			if (bdla_Mxf_value(A, i, j) != 0) {
				is_triu = 0;
				break;
			}
		}
		if (is_triu == 0) { break; }
	}
	return is_triu;
}

BDLA_EXPORT bdla_Status bdla_Mxf_zero(bdla_Mxf *A) {
	assert(A != NULL);
	assert(A->arr != NULL);
	memset(A->arr, 0x0, sizeof(float)*A->dims[1]*A->dims[0]);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_plus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[1] > 0);
	assert(Y->dims[0] > 0);
	assert(A.arr != NULL);
	assert(A.dims[1] > 0);
	assert(A.dims[0] > 0);
	assert(B.arr != NULL);
	assert(B.dims[1] > 0);
	assert(B.dims[0] > 0);

	int i, max;
	if (A.dims[1] != B.dims[1] || A.dims[0] != B.dims[0]) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) { 
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]); 
	}
	max = A.dims[1] * A.dims[0];
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
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] + b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_minus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[1] > 0);
	assert(Y->dims[0] > 0);
	assert(A.arr != NULL);
	assert(A.dims[1] > 0);
	assert(A.dims[0] > 0);
	assert(B.arr != NULL);
	assert(B.dims[1] > 0);
	assert(B.dims[0] > 0);

	int i, max;
	if (A.dims[1] != B.dims[1] && A.dims[0] != B.dims[0]) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.dims[1] != Y->dims[1] && A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	max = A.dims[1] * A.dims[0];
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
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] - b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_fmult(bdla_Mxf A, float b, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.dims[1] != Y->dims[1] && A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] * b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_ewmult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	if (A.dims[1] != B.dims[1] || A.dims[0] != B.dims[0]) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] * B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_mult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(B.arr != NULL);
	assert(B.dims[0] > 0);
	assert(B.dims[1] > 0);
	if (A.dims[1] != B.dims[0]) { return BDLA_DIMENSION_MISMATCH; }
	if (A.dims[0] != Y->dims[0] || B.dims[1] != Y->dims[1]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int alias = 0;
	float *outarr = Y->arr;
	if (Y->arr == A.arr) {
		alias = 1;
		outarr = malloc(sizeof(float) * Y->dims[1] * Y->dims[0]);
		if (outarr == NULL) { return BDLA_MEM_ERROR; }
	}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.dims[0],
		B.dims[1], B.dims[0], 1.f, A.arr, A.dims[1], B.arr, B.dims[1], 0.f, outarr, Y->dims[1]);
	if (alias) {
		Y->arr = outarr;
		free(A.arr);
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_mult_ext(bdla_Mxf A, bdla_MatrixProperty A_prop,
	bdla_Mxf B, bdla_MatrixProperty B_prop, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(B.arr != NULL);
	assert(B.dims[0] > 0);
	assert(B.dims[1] > 0);
	if (A.dims[1] != B.dims[0]) { return BDLA_DIMENSION_MISMATCH; }
	if (A.dims[0] != Y->dims[0] || B.dims[1] != Y->dims[1]) {
		bdla_Mxf_resize(Y, A.dims[0], B.dims[1]);
	}
	int alias = 0;
	float *outarr = Y->arr;
	if (Y->arr == A.arr) {
		alias = 1;
		outarr = malloc(sizeof(float) * Y->dims[1] * Y->dims[0]);
		if (outarr == NULL) { return BDLA_MEM_ERROR; }
	}
	if (Y->arr == B.arr) {
		alias = 2;
		outarr = malloc(sizeof(float) * Y->dims[1] * Y->dims[0]);
		if (outarr == NULL) { return BDLA_MEM_ERROR; }
	}
	/* Level 3 routines are either symmetric or tri. Assume tri is easier?
	trmm:	B = AB or B = BA where A is special.
	symm:	C = AB or C = BA where A is special
	*/
	if (A_prop == BDLA_MATRIX_TRI_UPPER || A_prop == BDLA_MATRIX_TRI_LOWER) {
		if (!bdla_Mxf_issquare(A)) { return BDLA_NONSQUARE; }
		memcpy(outarr, B.arr, sizeof(float) * Y->dims[1] * Y->dims[2]);
		if (A_prop == BDLA_MATRIX_TRI_LOWER) {
			cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
				B.dims[0], B.dims[1], 1.f, A.arr, A.dims[1], outarr, B.dims[1]);
		}
		if (A_prop == BDLA_MATRIX_TRI_UPPER) {
			cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				B.dims[0], B.dims[1], 1.f, A.arr, A.dims[1], outarr, B.dims[1]);
		}
	} else if (B_prop == BDLA_MATRIX_TRI_UPPER || B_prop == BDLA_MATRIX_TRI_LOWER) {
		if (!bdla_Mxf_issquare(B)) { return BDLA_NONSQUARE; }
		memcpy(outarr, A.arr, sizeof(float) * Y->dims[1] * Y->dims[2]);
		if (B_prop == BDLA_MATRIX_TRI_LOWER) {
			cblas_strmm(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
				B.dims[0], B.dims[1], 1.f, outarr, B.dims[1], A.arr, A.dims[1]);
		}
		if (B_prop == BDLA_MATRIX_TRI_UPPER) {
			cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
				B.dims[0], B.dims[1], 1.f, outarr, B.dims[1], A.arr, A.dims[1]);
		}
	}
	else if (A_prop == BDLA_MATRIX_SYMMETRIC) {
		if (!bdla_Mxf_issquare(A)) { return BDLA_NONSQUARE; }
		cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, Y->dims[0], Y->dims[1], 1.f,
			A.arr, A.dims[1], B.arr, B.dims[1], 0.f, outarr, Y->dims[1]);
	}
	else if (B_prop == BDLA_MATRIX_SYMMETRIC) {
		if (!bdla_Mxf_issquare(B)) { return BDLA_NONSQUARE; }
		cblas_ssymm(CblasRowMajor, CblasRight, CblasUpper, Y->dims[0], Y->dims[1], 1.f,
			B.arr, B.dims[1], A.arr, A.dims[1], 0.f, outarr, Y->dims[1]);
	}
	else { /* General matrix-matrix multiply. */
		Y->arr = outarr;
		bdla_Mxf_mult(A, B, Y);
	}
	if (alias) {
		Y->arr = outarr;
		if (alias == 1) {
			free(A.arr);
		}
		else if (alias == 2) {
			free(B.arr);
		}
		else {
			assert(0);
			return BDLA_MEM_ERROR;
		}
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_vmult(bdla_Mxf A, bdla_Vxf b, bdla_Vxf *y) {
	assert(A.arr != NULL);
	assert(A.dims[1] >= 0);
	assert(A.dims[0] >= 0);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	assert(b.len >= 0);
	assert(b.arr != NULL);
	if (A.dims[0] != y->len || A.dims[1] != b.len) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	int alias = 0;
	float *tmparr = y->arr;
	if (y->arr == b.arr) {
		alias = 1;
		tmparr = malloc(sizeof(float) * y->len);
	}
	cblas_sgemv(CblasRowMajor, CblasNoTrans, A.dims[0], A.dims[1], 1.f,
		A.arr, A.dims[1], b.arr, 1, 0.f, tmparr, 1);
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
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] / b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_ewdiv(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(B.arr != NULL);
	assert(B.dims[0] > 0);
	assert(B.dims[1] > 0);
	if (A.dims[1] != B.dims[1] || A.dims[0] != B.dims[0]) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	if (A.dims[1] != Y->dims[1] || A.dims[0] != Y->dims[0]) {
		bdla_Mxf_resize(Y, A.dims[0], A.dims[1]);
	}
	int i, max;
	max = A.dims[1] * A.dims[0];
	for (i = 0; i < max; ++i) {
		Y->arr[i] = A.arr[i] / B.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_trisolve(bdla_Mxf A, bdla_MatrixProperty A_prop,
	bdla_Mxf B, bdla_Mxf *Y) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(B.arr != NULL);
	assert(B.dims[0] > 0);
	assert(B.dims[1] > 0);
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[0] > 0);
	assert(Y->dims[1] > 0);
	assert(A_prop == BDLA_MATRIX_TRI_UPPER || A_prop == BDLA_MATRIX_TRI_LOWER);
	int alias = 0;
	if(A.arr == Y->arr){	/* Create buffers for aliased memory. */
		alias = 1;
		Y->arr = malloc(sizeof(float) * Y->dims[0] * Y->dims[1]);
		if (Y->arr == NULL) { return BDLA_MEM_ERROR; }
	}
	if (Y->dims[0] != B.dims[0] || Y->dims[1] != B.dims[1]) {
		if (bdla_Mxf_resize(Y, B.dims[0], B.dims[1]) != BDLA_GOOD) {
			return BDLA_MEM_ERROR;
		}
	}
	if (Y->arr != B.arr) {	/* strsm is an implace operation */
		memcpy(Y->arr, B.arr, sizeof(float) * B.dims[0] * B.dims[1]); 
	}
	if (A_prop == BDLA_MATRIX_TRI_LOWER) {
		cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
			Y->dims[0], Y->dims[1], 1.f, A.arr, A.dims[0], Y->arr, Y->dims[1]);
	} 
	else if (A_prop == BDLA_MATRIX_TRI_UPPER) {
		cblas_strsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
			Y->dims[0], Y->dims[1], 1.f, A.arr, A.dims[0], Y->arr, Y->dims[1]);
	}
	else { return BDLA_BAD_PROPERTY; }
	if (alias == 1) {		/* Free swapped buffer due to aliasing. */
		free(A.arr);
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_vtrisolve(bdla_Mxf A, bdla_MatrixProperty A_prop,
	bdla_Vxf b, bdla_Vxf *y) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(b.arr != NULL);
	assert(b.len > 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len > 0);
	assert(y->len == b.len);
	assert(A_prop == BDLA_MATRIX_TRI_UPPER || A_prop == BDLA_MATRIX_TRI_LOWER);
	int alias = 0;
	float *outarr = y->arr;
	if (y->arr == b.arr) {
		assert(y->len == b.len);
		alias = 1;
		outarr = malloc(sizeof(float) * y->len);
		if (outarr == NULL) { return BDLA_MEM_ERROR; }
	}
	memcpy(outarr, b.arr, sizeof(float) * b.len);
	if (A_prop == BDLA_MATRIX_TRI_UPPER) {
		cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 
			A.dims[0], A.arr, A.dims[1], outarr, 1);
	}
	else if(A_prop == BDLA_MATRIX_TRI_LOWER){
		cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
			A.dims[0], A.arr, A.dims[1], outarr, 1);
	}
	else {
		return BDLA_BAD_PROPERTY;
	}
	if (alias) {
		y->arr = outarr;
		free(b.arr);
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_diagsolve(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(B.arr != NULL);
	assert(B.dims[0] > 0);
	assert(B.dims[1] > 0);
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->dims[0] > 0);
	assert(Y->dims[1] > 0);
	if (!bdla_Mxf_issquare(A)) { return BDLA_NONSQUARE; }
	if (A.dims[1] != B.dims[0]) { return BDLA_DIMENSION_MISMATCH; }
	if (B.dims[0] != Y->dims[0] || B.dims[1] != Y->dims[0]) {
		if (bdla_Mxf_resize(Y, B.dims[0], B.dims[1]) != BDLA_GOOD) {
			return BDLA_MEM_ERROR;
		}
	}
	/* We don't need to worry about aliasing. */
	bdla_Vxf d = bdla_Vxf_create(A.dims[0]);
	int i, j;
	for (i = 0; i < A.dims[0]; ++i) {
		bdla_Vxf_writevalue(d, i, bdla_Mxf_value(A, i, i));
	}
	for (i = 0; i < B.dims[0]; ++i) {
		float mult = 1.f / bdla_Vxf_value(d, i);
		for (j = 0; j < B.dims[1]; ++j) {
			bdla_Mxf_writevalue(*Y, i, j, mult * bdla_Mxf_value(B, i, j));
		}
	}
	bdla_Vxf_release(&d);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_vdiagsolve(bdla_Mxf A, bdla_Vxf b, bdla_Vxf *y) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	assert(b.arr != NULL);
	assert(b.len > 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len > 0);
	assert(y->len == b.len);
	if (!bdla_Mxf_issquare(A)) { return BDLA_NONSQUARE; }
	if (A.dims[1] != b.len) { return BDLA_DIMENSION_MISMATCH; }
	if (b.len != y->len) {
		if (bdla_Vxf_resize(y, b.len) != BDLA_GOOD) {
			return BDLA_MEM_ERROR;
		}
	}
	/* We don't need to worry about aliasing. */
	int i;
	for (i = 0; i < A.dims[0]; ++i) {
		bdla_Vxf_writevalue(*y, i,
			bdla_Vxf_value(b, i) / bdla_Mxf_value(A, i, i));
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_row(bdla_Mxf A, int row, bdla_Vxf *y) {
	assert(y != NULL);	
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	if (A.dims[1] != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (row < 0 || row > A.dims[0]) { return BDLA_BAD_INDEX; }
	memcpy(y->arr, &A.arr[row * A.dims[1]], sizeof(float)*A.dims[1]);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writerow(bdla_Mxf A, int row, bdla_Vxf *y) {
	assert(y != NULL);	
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	if (A.dims[1] != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (row < 0 || row > A.dims[0]) { return BDLA_BAD_INDEX; }
	memcpy(&A.arr[row * A.dims[1]], y->arr, sizeof(float)*A.dims[1]);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_col(bdla_Mxf A, int col, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	if (A.dims[0] != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (col < 0 || col > A.dims[1]) { return BDLA_BAD_INDEX; }
	int i;
	for (i = 0; i < A.dims[0]; ++i) {
		y->arr[i] = A.arr[col + i * A.dims[1]];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writecol(bdla_Mxf A, int col, bdla_Vxf *y) {
	assert(y != NULL);
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[1] > 0);
	if (A.dims[0] != y->len) { return BDLA_DIMENSION_MISMATCH; }
	if (col < 0 || col > A.dims[1]) { return BDLA_BAD_INDEX; }
	int i;
	for (i = 0; i < A.dims[0]; ++i) {
		A.arr[col + i * A.dims[1]] = y->arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_submat(bdla_Mxf A, int row, int col, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr == NULL);
	assert(Y->dims[1] >= 0);
	assert(Y->dims[0] >= 0);
	assert(A.arr != NULL);
	assert(A.dims[0] >= 0);
	assert(A.dims[1] >= 0);
	if (row < 0 || col < 0) { return BDLA_BAD_INDEX; }
	if (A.dims[0] < row + Y->dims[0]) { return BDLA_BAD_INDEX; }
	if (A.dims[1] < col + Y->dims[1]) { return BDLA_BAD_INDEX; }
	int i, j, k, maxi, maxj;
	maxi = row + Y->dims[0];
	maxj = col + Y->dims[1];
	k = 0;
	for (i = row; i < maxi; ++i) {
		for (j = col; j < maxj; ++j) {
			Y->arr[k] = A.arr[i * A.dims[1] + j];
			++k;
		}
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_writesubmat(bdla_Mxf A, int row, int col, bdla_Mxf *Y) {
	assert(Y != NULL);
	assert(Y->arr == NULL);
	assert(Y->dims[1] >= 0);
	assert(Y->dims[0] >= 0);
	assert(A.arr != NULL);
	assert(A.dims[0] >= 0);
	assert(A.dims[1] >= 0);
	if (row < 0 || col < 0) { return BDLA_BAD_INDEX; }
	if (A.dims[0] < row + Y->dims[0]) { return BDLA_BAD_INDEX; }
	if (A.dims[1] < col + Y->dims[1]) { return BDLA_BAD_INDEX; }
	int i, j, k, maxi, maxj;
	maxi = row + Y->dims[0];
	maxj = col + Y->dims[1];
	k = 0;
	for (i = row; i < maxi; ++i) {
		for (j = col; j < maxj; ++j) {
			A.arr[i * A.dims[1] + j] = Y->arr[k];
			++k;
		}
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_uniform(bdla_Mxf *A, float b) {
	assert(A->arr != NULL);
	assert(A->dims[0] >= 0);
	assert(A->dims[1] >= 0);
	int i, maxi;
	maxi = A->dims[0] * A->dims[1];
	for (i = 0; i < maxi; ++i) {
		A->arr[i] = b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Mxf_eye(bdla_Mxf *A) {
	assert(A->arr != NULL);
	assert(A->dims[0] >= 0);
	assert(A->dims[1] >= 0);
	if (A->dims[0] != A->dims[1]) { return BDLA_NONSQUARE; }
	int i, maxi;
	maxi = A->dims[0] * A->dims[0];
	bdla_Status stat = BDLA_GOOD;
	stat = bdla_Mxf_zero(A);
	if (stat == BDLA_GOOD) {
		for (i = 0; i < maxi; i += A->dims[0] + 1 ) {
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
	assert(A->dims[0] >= 0);
	assert(A->dims[1] >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	if (A->dims[0] != A->dims[1]) { return BDLA_NONSQUARE; }
	if (A->dims[0] - abs(k) != b.len) { return BDLA_DIMENSION_MISMATCH; }
	int i, j, maxi;
	bdla_Status stat = BDLA_GOOD;
	i = k >= 0 ? k : abs(k)*A->dims[1];
	maxi = k >= 0 ? A->dims[0] * (A->dims[0] - abs(k)) : A->dims[0] * A->dims[0];
	stat = bdla_Mxf_zero(A);
	if (stat == BDLA_GOOD) {
		for (j = 0; i < maxi; i += A->dims[0] + 1, ++j) {
			A->arr[i] = b.arr[j];
		}
	}
	return stat;
}
