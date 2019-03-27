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
	/* Basic writing */
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
	bdla_Vxf_release(&a);
    return 0;
}
#endif /* BSV_TEST_VEC3F_H */
