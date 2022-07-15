/* ------------------------------------------------------

 View Frustum - Lighthouse3D

  -----------------------------------------------------*/


#ifndef _FRUSTUMG_
#define _FRUSTUMG_

#ifndef _VEC3_
#include "Vec3.h"
#endif

class Vec3;

#ifndef _PLANE_
#include "Plane.h"
#endif

class Plane;

#ifndef _AABOX_
#include "AABox.h"
#endif

class AABox;




class FrustumG 
{
private:

	enum {
		TOP = 0,
		BOTTOM,
		LEFT,
		RIGHT,
		NEARP,
		FARP
	};


public:

	static enum {OUTSIDE, INTERSECT, INSIDE};

	Plane pl[6];


	Vec3 ntl,ntr,nbl,nbr,ftl,ftr,fbl,fbr;
	float nearD, farD, ratio, angle,tang;
	float nw,nh,fw,fh;

	FrustumG::FrustumG();
	FrustumG::~FrustumG();

	void FrustumG::setCamInternals(float angle, float ratio, float nearD, float farD);
	void FrustumG::setCamDef(Vec3 &p, Vec3 &l, Vec3 &u);
	int FrustumG::pointInFrustum(Vec3 &p);
	int FrustumG::sphereInFrustum(Vec3 &p, float raio);
	int FrustumG::boxInFrustum(AABox &b);

	void FrustumG::drawPoints();
	void FrustumG::drawLines();
	void FrustumG::drawPlanes();
	void FrustumG::drawNormals();
};


#endif