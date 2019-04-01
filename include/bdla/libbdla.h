#ifndef BDLA_LIBBDLA_H
#define BDLA_LIBBDLA_H
/*============================================================================
libbdla.h

A dense linear algebra library in C

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
#ifndef BDLA_EXPORT
# ifdef _WIN32
#  define BDLA_EXPORT __declspec(dllimport)
# else
#  define BDLA_EXPORT
# endif
#endif 
#include <assert.h>

typedef struct {
	int dims[2];
	float *arr;
} bdla_Mxf;

typedef struct {
	int len;
	float *arr;
} bdla_Vxf;

typedef enum {
	BDLA_GOOD = 0,
	BDLA_DIMENSION_MISMATCH = -1,
	BDLA_MEM_ERROR = -2,
	BDLA_BAD_INDEX = -3,
	BDLA_UNDERSIZED = -4,
	BDLA_NONSQUARE = -5
} bdla_Status;

/* Creation & destruction */
BDLA_EXPORT bdla_Mxf bdla_Mxf_create(int r, int c);
BDLA_EXPORT void bdla_Mxf_release(bdla_Mxf *mat);
BDLA_EXPORT bdla_Mxf bdla_Mxf_copy(bdla_Mxf mat);
/* Info */
BDLA_EXPORT int bdla_Mxf_rows(bdla_Mxf A);
BDLA_EXPORT int bdla_Mxf_cols(bdla_Mxf A);
BDLA_EXPORT int bdla_Mxf_isequal(bdla_Mxf a, bdla_Mxf b);
/* Manipulation */
BDLA_EXPORT bdla_Status bdla_Mxf_fplus(bdla_Mxf A, float b, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_plus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_fminus(bdla_Mxf A, float B, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_minus(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_fmult(bdla_Mxf A, float b, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_ewmult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_mult(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_vmult(bdla_Mxf A, bdla_Vxf b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Mxf_fdiv(bdla_Mxf A, float b, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_ewdiv(bdla_Mxf A, bdla_Mxf B, bdla_Mxf *Y);
/* Writing and reading */
static inline float bdla_Mxf_value(bdla_Mxf A, int row, int col);
static inline void bdla_Mxf_writevalue(bdla_Mxf A, int row, int col, float y);
BDLA_EXPORT bdla_Status bdla_Mxf_row(bdla_Mxf A, int row, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Mxf_writerow(bdla_Mxf A, int row, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Mxf_col(bdla_Mxf A, int col, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Mxf_writecol(bdla_Mxf A, int col, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Mxf_submat(bdla_Mxf A, int row, int col, bdla_Mxf *Y);
BDLA_EXPORT bdla_Status bdla_Mxf_writesubmat(bdla_Mxf A, int row, int col, bdla_Mxf *Y);
/* Setting to specific values */
BDLA_EXPORT bdla_Status bdla_Mxf_zero(bdla_Mxf *A);
BDLA_EXPORT bdla_Status bdla_Mxf_uniform(bdla_Mxf *A, float b);
BDLA_EXPORT bdla_Status bdla_Mxf_eye(bdla_Mxf *A);
BDLA_EXPORT bdla_Status bdla_Mxf_diag(bdla_Mxf *A, bdla_Vxf b, int k);

/* Creation */
BDLA_EXPORT bdla_Vxf bdla_Vxf_create(int len);
BDLA_EXPORT void bdla_Vxf_release(bdla_Vxf *vec);
BDLA_EXPORT bdla_Vxf bdla_Vxf_copy(bdla_Vxf vec);
/* Info */
BDLA_EXPORT int bdla_Vxf_length(bdla_Vxf a);
BDLA_EXPORT int bdla_Vxf_isequal(bdla_Vxf a, bdla_Vxf b);
BDLA_EXPORT int bdla_Vxf_isfinite(bdla_Vxf a);
BDLA_EXPORT float bdla_Vxf_min(bdla_Vxf a);
BDLA_EXPORT float bdla_Vxf_max(bdla_Vxf a);
BDLA_EXPORT bdla_Status bdla_Vxf_minmax(bdla_Vxf a, float *min, float *max);
/* Functions */
BDLA_EXPORT bdla_Status bdla_Vxf_fplus(bdla_Vxf a, float b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_plus(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_fminus(bdla_Vxf a, float b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_minus(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_fmult(bdla_Vxf a, float b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_ewmult(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_outer(bdla_Vxf a, bdla_Vxf b, bdla_Mxf *Y);
BDLA_EXPORT float bdla_Vxf_dot(bdla_Vxf a, bdla_Vxf b);
BDLA_EXPORT float bdla_Vxf_norm2(bdla_Vxf a);
BDLA_EXPORT float bdla_Vxf_sum(bdla_Vxf a);
BDLA_EXPORT float bdla_Vxf_abssum(bdla_Vxf a);
/* Writing and reading */
static inline float bdla_Vxf_value(bdla_Vxf A, int pos);
static inline void bdla_Vxf_writevalue(bdla_Vxf A, int pos, float y);
BDLA_EXPORT bdla_Status bdla_Vxf_subvec(bdla_Vxf a, int pos, bdla_Vxf *y);
BDLA_EXPORT bdla_Status bdla_Vxf_writesubvec(bdla_Vxf a, int pos, bdla_Vxf *y);
/* Setting to specific values */
BDLA_EXPORT bdla_Status bdla_Vxf_zero(bdla_Vxf *a);
BDLA_EXPORT bdla_Status bdla_Vxf_uniform(bdla_Vxf *a, float b);
BDLA_EXPORT bdla_Status bdla_Vxf_linspace(bdla_Vxf *a, float startval, float endval);


/* IMPLEMENTATION ----------------------------------------------------------*/

/* Implementation of direct access functions - inlined. */
static inline float bdla_Mxf_value(bdla_Mxf A, int row, int col) {
	assert(A.arr != NULL && "Bad input matrix");
	assert(row >= 0 && row < A.dims[0] && "Bad row index");
	assert(col >= 0 && col < A.dims[1] && "Bad column index");
	return A.arr[col + row * A.dims[1]];
}

static inline void bdla_Mxf_writevalue(bdla_Mxf A, int row, int col, float y) {
	assert(A.arr != 0 && "Bad input matrix");
	assert(row >= 0 && row < A.dims[0] && "Bad row index");
	assert(col >= 0 && col < A.dims[1] && "Bad column index");
	A.arr[col + row * A.dims[1]] = y;
}

static inline float bdla_Vxf_value(bdla_Vxf A, int pos) {
	assert(A.arr != NULL && "Bad input matrix");
	assert(pos >= 0 && pos < A.len && "Bad index");
	return A.arr[pos];
}

static inline void bdla_Vxf_writevalue(bdla_Vxf A, int pos, float y) {
	assert(A.arr != 0 && "Bad input matrix");
	assert(pos >= 0 && pos < A.len && "Bad index");
	A.arr[pos] = y;
}


#endif /* BDLA_LIBBDLA_H */
