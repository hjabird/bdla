#ifndef BSV_TEST_VXF_H
#define BSV_TEST_VXF_H
/*============================================================================
test_blasVxF.h

Test functionality of float vectors.

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
#include "../include/bdla/libbdla.h"

#include <math.h>

int testVxf(){
    SECTION("Vector float");
	bdla_Vxf a, b, c, d;
	a = bdla_Vxf_create(4);
	/* Basic writing & writing */
	TEST(bdla_Vxf_length(a) == 4);
	TEST(bdla_Vxf_writevalue(a, 0, 1.f) == BDLA_GOOD);
	TEST(bdla_Vxf_writevalue(a, 1, 2.f) == BDLA_GOOD);
	TEST(bdla_Vxf_writevalue(a, 2, 3.f) == BDLA_GOOD);
	TEST(bdla_Vxf_writevalue(a, 3, 4.f) == BDLA_GOOD);
	TEST(bdla_Vxf_writevalue(a, 4, 5.f) == BDLA_BAD_INDEX);
	float out;
	TEST(bdla_Vxf_value(a, 2, &out) == BDLA_GOOD);
	TEST(out == 3.f);
	bdla_Vxf_value(a, 0, &out);
	TEST(out == 1.f);
	TEST(bdla_Vxf_value(a, -1, &out) == BDLA_BAD_INDEX);
	/* Copying */
	b = bdla_Vxf_copy(a);
	TEST(bdla_Vxf_length(b) == 4);
	TEST(bdla_Vxf_value(a, 2, &out) == BDLA_GOOD);
	TEST(out == 3.f);
	TEST(bdla_Vxf_isequal(a, b));

	/* Generating */			/* Uniform value */
	d = bdla_Vxf_create(10);
	bdla_Vxf_uniform(&d, 3.f);
	bdla_Vxf_value(d, 0, &out);
	TEST(out == 3.f);
	bdla_Vxf_value(d, 7, &out);
	TEST(out == 3.f);
	TEST(bdla_Vxf_value(d, 10, &out) == BDLA_BAD_INDEX);
	/* Generating */			/* Linear spacing */
	bdla_Vxf_linspace(&d, 1.f, 11.f);
	bdla_Vxf_value(d, 0, &out);
	TEST(out == 1.f);
	bdla_Vxf_value(d, 9, &out);
	TEST(out == 11.f);
	bdla_Vxf_value(d, 4, &out);
	TEST(out == 5.444444444444444444444444f);
	TEST(!bdla_Vxf_isequal(a, d));
	bdla_Vxf_zero(&d);
	bdla_Vxf_value(d, 4, &out);
	TEST(out == 0.f);

	/* Basic arithmatic. */		/* PLUS */
	c = bdla_Vxf_copy(a);
	bdla_Vxf_plus(a, b, &c);
	bdla_Vxf_value(c, 2, &out);
	TEST(out == 6.f);
	bdla_Vxf_plus(a, a, &a);	/* inplace */
	bdla_Vxf_value(a, 2, &out);
	TEST(out == 6.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 4.f);
	bdla_Vxf_plus(a, b, &a);	/* different inputs */
	bdla_Vxf_value(a, 2, &out);
	TEST(out == 9.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 6.f);
	bdla_Vxf_fplus(a, -3.f, &a);
	bdla_Vxf_value(a, 2, &out);
	TEST(out == 6.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 3.f);

	/* Basic arithmatic. */		/* MINUS */
	bdla_Vxf_linspace(&b, 0.f, 3.f);
	bdla_Vxf_uniform(&a, 1.f);
	TEST(bdla_Vxf_minus(b, a, &d) == BDLA_DIMENSION_MISMATCH);
	bdla_Vxf_minus(a, b, &c);
	bdla_Vxf_value(c, 2, &out);
	TEST(out == -1.f);
	bdla_Vxf_value(c, 1, &out);
	TEST(out == 0.f);
	TEST(bdla_Vxf_fminus(b, 2, &d) == BDLA_DIMENSION_MISMATCH);
	bdla_Vxf_fminus(b, -1.f, &a);
	bdla_Vxf_value(a, 2, &out);
	TEST(out == 3.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 2.f);

	/* Basic arithmatic. */		/* MULTIPLY */
	bdla_Vxf_linspace(&b, 0.f, 3.f);	/* by a scalar */
	TEST(bdla_Vxf_fmult(b, 2, &d) == BDLA_DIMENSION_MISMATCH);
	bdla_Vxf_fmult(b, 2, &a);
	bdla_Vxf_value(a, 2, &out);
	TEST(out == 4.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 2.f);
	bdla_Vxf_ewmult(b, b, &a);
	bdla_Vxf_value(a, 3, &out);
	TEST(out == 9.f);
	bdla_Vxf_value(a, 1, &out);
	TEST(out == 1.f);
	/* Bare in mind failures here may be due to bad matrices. */
	bdla_Mxf mat = bdla_Mxf_create(4, 10);
	bdla_Vxf_linspace(&d, 1.f, 10.f);
	bdla_Vxf_linspace(&b, 1.f, 4.f);
	bdla_Vxf_outer(b, d, &mat);
	bdla_Mxf_value(mat, 2, 2, &out);
	TEST(out == 9.f);
	bdla_Mxf_value(mat, 3, 4, &out);
	TEST(out == 20.f);
	bdla_Mxf_value(mat, 1, 8, &out);
	TEST(out == 18.f);

	/* Dot product */
	bdla_Vxf_linspace(&b, 1.f, 4.f);
	bdla_Vxf_linspace(&a, 0.f, 3.f);
	bdla_Vxf_dot(a, b, &out);
	TEST(out == 20.f);
	/* Norm2 */
	out = bdla_Vxf_norm2(a);
	TEST(out == sqrtf(14.f));
	/* Sum */
	TEST(bdla_Vxf_sum(a) == 6.f);
	TEST(bdla_Vxf_abssum(a) == 6.f);
	bdla_Vxf_linspace(&b, -2.f, 1.f);
	TEST(bdla_Vxf_abssum(b) == 4.f);

	/* Min & max */
	bdla_Vxf_linspace(&b, 1.f, 4.f);
	bdla_Vxf_linspace(&a, 0.f, 3.f);
	TEST(bdla_Vxf_min(b) == 1.f);
	TEST(bdla_Vxf_max(b) == 4.f);
	TEST(bdla_Vxf_min(a) == 0.f);
	TEST(bdla_Vxf_max(a) == 3.f);
	float min, max;
	bdla_Vxf_minmax(a, &min, &max);
	TEST(min == 0.);
	TEST(max == 3.);

	bdla_Vxf_release(&b);
	bdla_Vxf_release(&a);
	bdla_Vxf_release(&c);
	bdla_Vxf_release(&d);
    return 0;
}
#endif /* BSV_TEST_VEC3F_H */
