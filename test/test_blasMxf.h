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
	TEST(bdla_Mxf_writevalue(a, 1, 1, 3.f) == BDLA_GOOD);
	TEST(bdla_Mxf_writevalue(a, 3, 2, 0.f) == BDLA_BAD_INDEX);
	float out;
	TEST(bdla_Mxf_value(a, 1, 1, &out) == BDLA_GOOD);
	TEST(out == 3.f);

	/* Filling with values... */	/* uniform */
	bdla_Mxf_uniform(&a, 1.f);		
	bdla_Mxf_value(a, 1, 1, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(a, 2, 0, &out);
	TEST(out == 1.f);
	/* Filling with values... */	/* zero */
	bdla_Mxf_zero(&b);				
	bdla_Mxf_value(b, 1, 1, &out);
	TEST(out == 0.f);
	bdla_Mxf_value(b, 2, 0, &out);
	TEST(out == 0.f);
	TEST(!bdla_Mxf_isequal(a, b));
	bdla_Mxf_release(&c);
	c = bdla_Mxf_create(5, 5);
	/* Filling with values... */	/* eye */
	bdla_Mxf_eye(&c);				
	bdla_Mxf_value(c, 1, 1, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(c, 2, 2, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(c, 4, 4, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(c, 1, 4, &out);
	TEST(out == 0.f);
	bdla_Mxf_value(c, 0, 2, &out);
	TEST(out == 0.f);
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
	bdla_Mxf_value(b, 0, 2, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(b, 1, 4, &out);
	TEST(out == 0.f);
	bdla_Mxf_value(b, 0, 3, &out);
	TEST(out == 0.f);

	/* Operations */				/* scalar plus */
	bdla_Mxf_release(&a);
	bdla_Mxf_release(&b);
	bdla_Mxf_release(&c);
	a = bdla_Mxf_create(3, 4);
	b = bdla_Mxf_create(3, 4);
	c = bdla_Mxf_create(3, 4);
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_fplus(a, 2.f, &b);
	bdla_Mxf_value(b, 0, 2, &out);
	TEST(out == 4.f);
	bdla_Mxf_value(b, 1, 4, &out);
	TEST(out == 4.f);
	bdla_Mxf_value(b, 0, 3, &out);
	TEST(out == 4.f);
	bdla_Mxf_value(a, 0, 3, &out);
	TEST(out == 2.f);
	bdla_Mxf_fplus(a, 2.f, &a);
	bdla_Mxf_value(a, 0, 3, &out);
	TEST(out == 4.f);
	/* Operations */				/* plus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_uniform(&b, -1.f);
	bdla_Mxf_writevalue(b, 1, 2, 3.f);
	bdla_Mxf_plus(a, b, &c);
	bdla_Mxf_value(c, 1, 2, &out);
	TEST(out == 5.f);
	bdla_Mxf_value(c, 0, 3, &out);
	TEST(out == 1.f);
	/* Operations */				/* scalar minus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_fminus(a, 1.f, &b);
	bdla_Mxf_value(b, 0, 2, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(b, 1, 4, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(b, 0, 3, &out);
	TEST(out == 1.f);
	bdla_Mxf_value(b, 1, 2, &out);
	TEST(out == 2.f);
	/* Operations */				/* minus */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_uniform(&b, -1.f);
	bdla_Mxf_writevalue(b, 1, 2, 3.f);
	bdla_Mxf_minus(a, b, &c);
	bdla_Mxf_value(c, 1, 2, &out);
	TEST(out == -1.f);
	bdla_Mxf_value(c, 0, 3, &out);
	TEST(out == 3.f);
	/* Operations */				/* scalar mult */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_fmult(a, 2.f, &c);
	bdla_Mxf_value(c, 1, 2, &out);
	TEST(out == 6.f);
	bdla_Mxf_value(c, 0, 3, &out);
	TEST(out == 4.f);
	/* Operations */				/* elementwise mult */
	bdla_Mxf_uniform(&a, 2.f);
	bdla_Mxf_writevalue(a, 1, 2, 3.f);
	bdla_Mxf_uniform(&b, -4.f);
	bdla_Mxf_writevalue(b, 2, 0, 3.f);
	bdla_Mxf_ewmult(a, b, &c);
	bdla_Mxf_value(c, 1, 2, &out);
	TEST(out == -12.f);
	bdla_Mxf_value(c, 0, 3, &out);
	TEST(out == -8.f);
	bdla_Mxf_value(c, 2, 0, &out);
	TEST(out == 6.f);






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
