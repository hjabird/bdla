#ifndef BSV_TEST_MXF_H
#define BSV_TEST_MXF_H
/*============================================================================
test_blasMxF.h

Test functionality of float matrices.

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

void testMxf(){
	SECTION("Matrix float");
	bdla_Mxf a, b, c, d;
	bdla_Vxf va, vb, vc, vd;
	float out;

	/* Creation and copying */
	a = bdla_Mxf_create(3, 4);
	TEST(bdla_Mxf_rows(a) == 3);
	TEST(bdla_Mxf_cols(a) == 4);
	b = bdla_Mxf_copy(a);
	TEST(bdla_Mxf_rows(b) == 3);
	TEST(bdla_Mxf_cols(b) == 4);
	TEST(bdla_Mxf_isequal(a, b));
	c = bdla_Mxf_create(4, 3);
	d = bdla_Mxf_create(3, 4);
	TEST(!bdla_Mxf_isequal(a, c));

	/* Writing and reading */
	bdla_Mxf_writevalue(a, 1, 1, 3.f);
	TEST(bdla_Mxf_value(a, 1, 1) == 3.f);

	/* Filling with values... */	/* uniform */
	bdla_Mxf_uniform(&a, 1.f);		
	TEST(bdla_Mxf_value(a, 1, 1) == 1.f);
	TEST(bdla_Mxf_value(a, 2, 0) == 1.f);
	/* Filling with values... */	/* zero */
	bdla_Mxf_zero(&b);				
	TEST(bdla_Mxf_value(b, 1, 1) == 0.f);
	TEST(bdla_Mxf_value(b, 2, 0) == 0.f);
	TEST(!bdla_Mxf_isequal(a, b));
	bdla_Mxf_release(&c);
	c = bdla_Mxf_create(5, 5);
	/* Filling with values... */	/* eye */
	bdla_Mxf_eye(&c);				
	TEST(bdla_Mxf_value(c, 1, 1) == 1.f);
	TEST(bdla_Mxf_value(c, 1, 1) == 1.f);
	TEST(bdla_Mxf_value(c, 4, 4) == 1.f);
	TEST(bdla_Mxf_value(c, 1, 4) == 0.f);
	TEST(bdla_Mxf_value(c, 0, 2) == 0.f);
	/* Filling with values... */	/* diag */
	bdla_Mxf_release(&b);			
	b = bdla_Mxf_create(5, 5);
	va = bdla_Vxf_create(5);
	bdla_Vxf_uniform(&va, 1.f);
	bdla_Mxf_diag(&b, va, 0);
	TEST(bdla_Mxf_isequal(c, b));
	bdla_Vxf_release(&va);
	va = bdla_Vxf_create(3);
	bdla_Vxf_uniform(&va, 1.f);
	bdla_Mxf_diag(&b, va, 2);
	TEST(bdla_Mxf_value(b, 0, 2) == 1.f);
	TEST(bdla_Mxf_value(b, 1, 4) == 0.f);
	TEST(bdla_Mxf_value(b, 0, 3) == 0.f);

	/* Operations */				/* scalar plus */
	bdla_Mxf_release(&a);
	bdla_Mxf_release(&b);
	bdla_Mxf_release(&c);
	a = bdla_Mxf_create(3, 4);
	b = bdla_Mxf_create(3, 4);
	c = bdla_Mxf_create(3, 4);
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_fplus(a, 2.f, &b);
	TEST(bdla_Mxf_value(b, 0, 2) == 4.f);
	TEST(bdla_Mxf_value(b, 1, 2) == 4.f);
	TEST(bdla_Mxf_value(b, 0, 3) == 4.f);
	TEST(bdla_Mxf_value(a, 0, 3) == 2.f);
	bdla_Mxf_fplus(a, 2.f, &a);
	TEST(bdla_Mxf_value(a, 0, 3) == 4.f);
	/* Operations */				/* plus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_uniform(&b, -1.f);
	bdla_Mxf_writevalue(b, 1, 2, 3.f);
	bdla_Mxf_plus(a, b, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == 5.f);
	TEST(bdla_Mxf_value(c, 0, 3) == 1.f);
	/* Operations */				/* scalar minus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_fminus(a, 1.f, &b);
	TEST(bdla_Mxf_value(b, 0, 2) == 1.f);
	TEST(bdla_Mxf_value(b, 1, 3) == 1.f);
	TEST(bdla_Mxf_value(b, 0, 3) == 1.f);
	TEST(bdla_Mxf_value(b, 1, 2) == 2.f);
	/* Operations */				/* minus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_uniform(&b, -1.f);
	bdla_Mxf_writevalue(b, 1, 2, 3.f);
	bdla_Mxf_minus(a, b, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == -1.f);
	TEST(bdla_Mxf_value(c, 0, 3) == 3.f);
	/* Operations */				/* scalar mult */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_fmult(a, 2.f, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == 6.f);
	TEST(bdla_Mxf_value(c, 0, 3) == 4.f);
	/* Operations */				/* elementwise mult */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_uniform(&b, -4.f);
	bdla_Mxf_writevalue(b, 2, 0, 3.f);
	bdla_Mxf_ewmult(a, b, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == -12.f);
	TEST(bdla_Mxf_value(c, 0, 3) == -8.f);
	TEST(bdla_Mxf_value(c, 2, 0) == 6.f);
	/* Operation */					/* general matrix mult */
	bdla_Mxf_release(&a);
	a = bdla_Mxf_create(3, 3);
	bdla_Mxf_eye(&a);	/* Identity matrix mult */
	TEST(bdla_Mxf_mult(a, b, &d) == BDLA_GOOD);
	TEST(bdla_Mxf_isequal(b, d));
	TEST(bdla_Mxf_mult(b, c, &d) == BDLA_DIMENSION_MISMATCH);
	/* And general multiplication.*/
	bdla_Mxf_writevalue(a, 0, 2, 3.f);
	bdla_Mxf_writevalue(a, 1, 2, 1.f);
	bdla_Mxf_writevalue(a, 2, 2, 4.f);
	bdla_Mxf_uniform(&b, 3.f);
	bdla_Mxf_writevalue(b, 1, 2, -4.f);
	bdla_Mxf_writevalue(b, 2, 2, 1.f);
	TEST(bdla_Mxf_mult(a, b, &d) == BDLA_GOOD);
	TEST(bdla_Mxf_value(d, 0, 0) == 12.f);
	TEST(bdla_Mxf_value(d, 0, 1) == 12.f);
	TEST(bdla_Mxf_value(d, 0, 2) == 6.f);
	TEST(bdla_Mxf_value(d, 0, 3) == 12.f);
	TEST(bdla_Mxf_value(d, 1, 2) == -3.f);
	TEST(bdla_Mxf_value(d, 2, 0) == 12.f);
	TEST(bdla_Mxf_value(d, 2, 2) == 4.f);
	/* Operation */				/* General matrix -vector multiplication */
	bdla_Vxf_release(&va);
	va = bdla_Vxf_create(3);
	bdla_Vxf_uniform(&va, 2.f);
	bdla_Vxf_writevalue(va, 2, 5.f);
	bdla_Mxf_vmult(a, va, &va);
	TEST(bdla_Vxf_value(va, 0) == 17.f);
	TEST(bdla_Vxf_value(va, 1) == 7.f);
	TEST(bdla_Vxf_value(va, 2) == 20.f);
	/* Operations */				/* scalar div */	
	bdla_Mxf_release(&a);
	bdla_Mxf_release(&b);
	bdla_Mxf_release(&c);
	a = bdla_Mxf_create(3, 4);
	b = bdla_Mxf_create(3, 4);
	c = bdla_Mxf_create(3, 4);
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_fdiv(a, 2.f, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == 1.5f);
	TEST(bdla_Mxf_value(c, 0, 3) == 1.f);
	/* Operations */				/* elementwise dov */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_uniform(&b, -4.f);
	bdla_Mxf_writevalue(b, 2, 0, 3.f);
	bdla_Mxf_ewdiv(a, b, &c);
	TEST(bdla_Mxf_value(c, 1, 2) == -0.75f);
	TEST(bdla_Mxf_value(c, 0, 3) == -0.5f);
	TEST(bdla_Mxf_value(c, 2, 0) == 2.f / 3.f);






	bdla_Mxf_release(&a);
	bdla_Mxf_release(&b);
	bdla_Mxf_release(&c);
	bdla_Mxf_release(&d);
	bdla_Vxf_release(&va);
	//bdla_Vxf_release(&vb);
	//bdla_Vxf_release(&vc);
	//bdla_Vxf_release(&vd);
}
#endif /* BSV_TEST_MXF_H */
