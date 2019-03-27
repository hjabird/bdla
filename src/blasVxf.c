#include "libbdla.h"
/*============================================================================
blasVxf.c

Floating point vector basic linear algebra.

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

#include <cblas.h>

BDLA_EXPORT bdla_Vxf bdla_Vxf_create(int len) {
	assert(len > 0);
	bdla_Vxf r = { len, malloc(sizeof(float)*len) };
	return r;
}

BDLA_EXPORT void bdla_Vxf_release(bdla_Vxf *mat) {
	if (mat != NULL) {
		assert(mat->arr != NULL);
		free(mat->arr); mat->arr = NULL;
		mat->len = 0;
	}
	return;
}

BDLA_EXPORT bdla_Vxf bdla_Vxf_copy(bdla_Vxf vec) {
	assert(vec.arr != NULL);
	bdla_Vxf ret = vec;
	ret.arr = malloc(sizeof(float) * ret.len);
	memcpy(ret.arr, vec.arr, sizeof(float) * ret.len);
	return ret;
}

BDLA_EXPORT bdla_Status bdla_Vxf_fplus(bdla_Vxf a, float b, bdla_Vxf *y){
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] + b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_plus(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len || a.len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] + b.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_fminus(bdla_Vxf a, float b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] - b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_minus(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len || a.len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] - b.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_fmult(bdla_Vxf a, float b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] * b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_ewmult(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len || a.len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] * b.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_fdiv(bdla_Vxf a, float b, bdla_Vxf *y){
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] / b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_ewdiv(bdla_Vxf a, bdla_Vxf b, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	int i;
	if (a.len != y->len || a.len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	for (i = 0; i < a.len; ++i) {
		y->arr[i] = a.arr[i] / b.arr[i];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_outer(bdla_Vxf a, bdla_Vxf b, bdla_Mxf *Y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(Y != NULL);
	assert(Y->arr != NULL);
	assert(Y->cols >= 0);
	assert(Y->rows >= 0);
	if (a.len != Y->rows || Y->cols != b.len ) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	cblas_sger(CblasRowMajor, a.len, b.len, 1.f, a.arr, 1, 
		b.arr, 1, Y->arr, Y->cols);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_dot(bdla_Vxf a, bdla_Vxf b, float *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	if (a.len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	*y = cblas_sdot(a.len, a.arr, 1, b.arr, 1);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_norm2(bdla_Vxf a, bdla_Vxf b, float *y) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	*y = cblas_snrm2(a.len, a.arr, 1);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_value(bdla_Vxf a, int pos, float *y) {
	assert(a.arr != NULL);
	assert(y != NULL);
	if (pos < 0 || pos >= a.len) { return BDLA_BAD_INDEX; }
	*y = a.arr[pos];
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_writevalue(bdla_Vxf a, int pos, float y) {
	assert(a.arr != NULL);
	if (pos < 0 || pos >= a.len) { return BDLA_BAD_INDEX; }
	a.arr[pos] = y;
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_subvec(bdla_Vxf a, int pos, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	if (pos < 0 || pos + y->len >= a.len) {
		return BDLA_BAD_INDEX;
	}
	int i;
	for(i = 0; i < y->len; ++i){
		y->arr[i] = a.arr[i + pos];
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_writesubvec(
	bdla_Vxf a, int pos, bdla_Vxf *y) {
	assert(a.arr != NULL);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(y->len >= 0);
	if (pos < 0 || pos + y->len >= a.len) {
		return BDLA_BAD_INDEX;
	}
	int i;
	for (i = 0; i < y->len; ++i) {
		a.arr[i + pos] = y->arr[i];
	}
	return BDLA_GOOD;
}
