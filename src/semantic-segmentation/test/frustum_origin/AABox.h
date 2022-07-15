/* ------------------------------------------------------

 Axis Aligned Boxes - Lighthouse3D

  -----------------------------------------------------*/


#ifndef _AABOX_
#define _AABOX_

#ifndef _Vec3_
#include "Vec3.h"
#endif

class Vec3;

class AABox 
{

public:

	Vec3 corner;
	float x,y,z;


	AABox::AABox( Vec3 &corner, float x, float y, float z);
	AABox::AABox(void);
	AABox::~AABox();

	void AABox::setBox( Vec3 &corner, float x, float y, float z);

	// for use in frustum computations
	Vec3 AABox::getVertexP(Vec3 &normal);
	Vec3 AABox::getVertexN(Vec3 &normal);


};


#endif