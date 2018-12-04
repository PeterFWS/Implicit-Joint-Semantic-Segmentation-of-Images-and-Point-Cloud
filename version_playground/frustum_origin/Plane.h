	// Plane.h
//
//////////////////////////////////////////////////////////////////////


#ifndef _PLANE_
#define _PLANE_

#ifndef _Vec3_
#include "Vec3.h"
#endif

class Vec3;

class Plane  
{

public:

	Vec3 normal,point;
	float d;


	Plane::Plane( Vec3 &v1,  Vec3 &v2,  Vec3 &v3);
	Plane::Plane(void);
	Plane::~Plane();

	void set3Points( Vec3 &v1,  Vec3 &v2,  Vec3 &v3);
	void setNormalAndPoint(Vec3 &normal, Vec3 &point);
	void setCoefficients(float a, float b, float c, float d);
	float distance(Vec3 &p);

	void print();

};


#endif