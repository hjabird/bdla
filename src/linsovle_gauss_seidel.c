#include "libbdla.h"
/*============================================================================
linsolve_gauss_seidel.c

Gauss-Seidel iterative method.

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

BDLA_EXPORT bdla_Status bdla_Mxf_solve_gauss_seidel(
	bdla_Mxf A, bdla_Vxf b, bdla_Vxf *y, float tol, bdla_Vxf *guess, int *max_iter) {
	assert(A.arr != NULL);
	assert(A.dims[0] > 0);
	assert(A.dims[0] > 0);
	assert(b.arr != NULL);
	assert(b.len >= 0);
	assert(y != NULL);
	assert(y->arr != NULL);
	assert(tol != 0.f);
	assert(guess != NULL ? (guess->arr != NULL && guess->len > 0) : 1);
	assert(max_iter != NULL ? *max_iter > 0 : 1);
	/* Check shapes */
	if (!bdla_Mxf_issquare(A)) { return BDLA_NONSQUARE; }
	if (b.len != A.dims[0]) { return BDLA_DIMENSION_MISMATCH; }
	if (guess != NULL && guess->len != b.len) { return BDLA_DIMENSION_MISMATCH; }
	if (tol > 1.f) { tol = 1e-6; }

	bdla_Vxf x;
	if (guess != NULL) {
		x = bdla_Vxf_copy(*guess);
	}
	else {
		x = bdla_Vxf_create(b.len);
		bdla_Vxf_zero(&x);
	}
	bdla_Mxf Ls = bdla_Mxf_create(A.dims[0], A.dims[1]);	/* Lower & diag matrix*/
	bdla_Mxf U = bdla_Mxf_create(A.dims[0], A.dims[1]);		/* Upper matrix*/
	bdla_Mxf_tri(A, 0, BDLA_MATRIX_TRI_LOWER, &Ls);
	bdla_Mxf_tri(A, 1, BDLA_MATRIX_TRI_UPPER, &U);
	bdla_Vxf diff = bdla_Vxf_create(A.dims[0]);
	float relerror = 9999999999.f;
	float bnorm = bdla_Vxf_norm2(b);
	int iter = 0;

	do {
		bdla_Mxf_vmult(U, x, &x);
		bdla_Vxf_minus(b, x, &x);
		bdla_Mxf_vtrisolve(Ls, BDLA_MATRIX_TRI_LOWER, x, &x);
		bdla_Mxf_vmult(A, x, &diff);
		bdla_Vxf_minus(diff, b, &diff);
		relerror = bdla_Vxf_norm2(diff) / bnorm;
		if (max_iter != NULL && iter >= *max_iter) { break; }
		++iter;
	} while (relerror > tol);

	bdla_Vxf_copyin(y, x);
	bdla_Vxf_release(&x);
	bdla_Vxf_release(&diff);
	bdla_Mxf_release(&Ls);
	bdla_Mxf_release(&U);
	return BDLA_GOOD;
}
