#include "../include/bdla/libbdla.h"

void testJacobi(){
	SECTION("Jacobi iterative solver");
	int sx = 4;
	bdla_Mxf mat = bdla_Mxf_create(sx, sx);
	bdla_Vxf a, b, c, d;
	a = bdla_Vxf_create(sx);
	b = bdla_Vxf_create(sx);
	c = bdla_Vxf_create(sx);
	d = bdla_Vxf_create(sx);
	bdla_Mxf_uniform(&mat, -1.f);
	bdla_Mxf_writevalue(mat, 0, 0, 10.f);
	bdla_Mxf_writevalue(mat, 1, 1, 11.f);
	bdla_Mxf_writevalue(mat, 2, 2, 10.f);
	bdla_Mxf_writevalue(mat, 3, 3, 8.f);
	bdla_Mxf_writevalue(mat, 0, 2, 2.f);
	bdla_Mxf_writevalue(mat, 0, 3, 0.f);
	bdla_Mxf_writevalue(mat, 1, 3, 3.f);
	bdla_Mxf_writevalue(mat, 2, 0, 2.f);
	bdla_Mxf_writevalue(mat, 3, 0, 0.f);
	bdla_Mxf_writevalue(mat, 3, 1, 3.f);
	bdla_Vxf_writevalue(a, 0, 6.f);
	bdla_Vxf_writevalue(a, 1, 25.f);
	bdla_Vxf_writevalue(a, 2, -11.f);
	bdla_Vxf_writevalue(a, 3, 15.f);
	bdla_Vxf_writevalue(d, 0, 1.f);
	bdla_Vxf_writevalue(d, 1, 2.f);
	bdla_Vxf_writevalue(d, 2, -1.f);
	bdla_Vxf_writevalue(d, 3, 1.f);
	bdla_Mxf_vmult(mat, a, &b);
	bdla_Mxf_solve_jacobi(mat, a, &c, 0.0000001f, NULL, NULL);
	bdla_Vxf_minus(c, d, &b);
	TEST(bdla_Vxf_norm2(b) / bdla_Vxf_norm2(d) < 0.0000001f);
}
