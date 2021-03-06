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

#include <openblas/cblas.h>
#include "nanimpl.h"

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

BDLA_EXPORT bdla_Status bdla_Vxf_resize(bdla_Vxf *a, int len) {
	assert(len > 0);
	assert(a != NULL);
	assert(a->arr != NULL);
	assert(a->len > 0);
	if (len != a->len) {
		a->arr = realloc(a->arr, sizeof(float) * len);
		a->len = len;
	}
	if (a->arr == NULL) { return BDLA_MEM_ERROR; }
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_copyin(bdla_Vxf *dest, bdla_Vxf source) {
	assert(dest != NULL);
	assert(dest->arr != NULL);
	assert(dest->len > 0);
	assert(source.arr != NULL);
	assert(source.len > 0);
	if (dest->arr == source.arr) { return BDLA_GOOD; }
	if (source.len != dest->len) {
		if (bdla_Vxf_resize(dest, source.len) != BDLA_GOOD) {
			return BDLA_MEM_ERROR;
		}
	}
	memcpy(dest->arr, source.arr, sizeof(float)*source.len);
	return BDLA_GOOD;
}

BDLA_EXPORT int bdla_Vxf_length(bdla_Vxf A) {
	assert(A.arr != NULL);
	assert(A.arr >= 0);
	return A.len;
}

BDLA_EXPORT int bdla_Vxf_isequal(bdla_Vxf a, bdla_Vxf b) {
	assert(a.arr != NULL);
	assert(b.arr != NULL);
	assert(a.len >= 0);
	assert(b.len >= 0);
	if (a.len != b.len) { return 0; }
	else {
		if(!memcmp(a.arr, b.arr, sizeof(float)*a.len)){ return 1; }
		else{	return 0;	}
	}
}

BDLA_EXPORT int bdla_Vxf_isfinite(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	int finite = 1, i;
	for(i = 0; i < a.len; ++i){
		if (a.arr[i] != a.arr[i]) { finite = 0; break; }
	}
	return finite;
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
	assert(Y->dims[1] >= 0);
	assert(Y->dims[0] >= 0);
	if (a.len != Y->dims[0] || Y->dims[1] != b.len ) { 
		return BDLA_DIMENSION_MISMATCH; 
	}
	/* Since we want to overwrite whatever is in Y. */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		a.len, b.len, 1, 1.f, a.arr, 1, b.arr, b.len, 0.f, Y->arr, b.len);
	return BDLA_GOOD;
}

BDLA_EXPORT float bdla_Vxf_dot(bdla_Vxf a, bdla_Vxf b) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(a.len == b.len);
	float y = cblas_sdot(a.len > b.len ? b.len : a.len, a.arr, 1, b.arr, 1);
	return y;
}

BDLA_EXPORT float bdla_Vxf_norm2(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	float y = cblas_snrm2(a.len, a.arr, 1);
	return y;
}

BDLA_EXPORT float bdla_Vxf_sum(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	int i;
	double acc = 0;
	for (i = 0; i < a.len; ++i) {
		acc += a.arr[i];
	}
	return (float)acc;
}

BDLA_EXPORT float bdla_Vxf_abssum(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 0);
	float y = cblas_sasum(a.len, a.arr, 1);
	return y;
}

BDLA_EXPORT float bdla_Vxf_min(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 1);
	float min;
	int i;
	if (a.len == 0) { min = gennanf(); }
	else {
		min = a.arr[0];
		for (i = 1; i < a.len; ++i) {
			min = a.arr[i] < min ? a.arr[i] : min;
		}
	}
	return min;
}

BDLA_EXPORT float bdla_Vxf_max(bdla_Vxf a) {
	assert(a.arr != NULL);
	assert(a.len >= 1);
	float max;
	int i;
	if (a.len == 0) { max = gennanf(); }
	else {
		max = a.arr[0];
		for (i = 1; i < a.len; ++i) {
			max = a.arr[i] > max ? a.arr[i] : max;
		}
	}
	return max;
}

BDLA_EXPORT bdla_Status bdla_Vxf_minmax(bdla_Vxf a, float *min, float *max) {
	assert(a.arr != NULL);
	assert(a.len >= 1);
	if (min == NULL && max == NULL) { return BDLA_GOOD; }
	if (max == NULL) {
		*min = bdla_Vxf_min(a);
	}
	else if (min == NULL) {
		*max = bdla_Vxf_max(a);
	}
	else {
		float lmin, lmax;
		int i;
		if (a.len == 0) { lmin = lmax = gennanf(); }
		else {
			lmax = lmin = a.arr[0];
			for (i = 1; i < a.len; ++i) {
				lmax = a.arr[i] > lmax ? a.arr[i] : lmax;
				lmin = a.arr[i] < lmin ? a.arr[i] : lmin;
			}
		}
		*min = lmin;
		*max = lmax;
	}
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

BDLA_EXPORT bdla_Status bdla_Vxf_zero(bdla_Vxf *a) {
	assert(a != NULL);
	assert(a->len >= 0);
	assert(a->arr != NULL);
	memset(a->arr, 0x00, sizeof(float) * a->len);
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_uniform(bdla_Vxf *a, float b) {
	assert(a != NULL);
	assert(a->len >= 0);
	assert(a->arr != NULL);
	int i;
	for (i = 0; i < a->len; ++i) {
		a->arr[i] = b;
	}
	return BDLA_GOOD;
}

BDLA_EXPORT bdla_Status bdla_Vxf_linspace(bdla_Vxf *a, float startval, float endval) {
	assert(a != NULL);
	assert(a->arr != NULL);
	assert(a->len >= 0);
	double interval;
	int i;
	if (a->len < 2) { return BDLA_UNDERSIZED; }
	interval = (double)(endval - startval) / (double)(a->len-1);
	for (i = 0; i < a->len; ++i) {
		a->arr[i] = (float)( startval + i * interval );
	}
	return BDLA_GOOD;
}
