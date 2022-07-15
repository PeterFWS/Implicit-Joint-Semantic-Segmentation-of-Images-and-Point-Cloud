/* ------------------------------------------------------

 View Frustum Culling - Lighthouse3D

  -----------------------------------------------------*/

#include <GL/glut.h>

#include "Vec3.h"
#include "FrustumG.h"

#include <stdio.h>

float a = 0;

float nearP = 1.0f, farP = 100.0f;
float angle = 45, ratio=1;

int frame=0, timebase=0;

int frustumOn = 1;
int spheresDrawn = 0;
int spheresTotal = 0;



FrustumG frustum;



void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	ratio = w * 1.0/ h;

	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	// Set the viewport to be the entire window
    glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(angle,ratio,nearP,farP);
	glMatrixMode(GL_MODELVIEW);

	frustum.setCamInternals(angle,ratio,nearP,farP);


}



void render() {

	glColor3f(0,1,0);
	spheresTotal=0;
	spheresDrawn=0;

	for (int i = -200; i < 200; i+=4) 
			for(int k =  -200; k < 200; k+=4) {
				spheresTotal++;
				Vec3 a(i,0,k);
				if (!frustumOn || (frustum.sphereInFrustum(a,0.5) != FrustumG::OUTSIDE)) {
					glPushMatrix();
					glTranslatef(i,0,k);
					glutSolidSphere(0.5,5,5);
					glPopMatrix();
					spheresDrawn++;
				}
			}

}


Vec3 p(0,0,5),l(0,0,0),u(0,1,0);

void renderScene(void) {

	char title[80];
	float fps,time;


	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(p.x,p.y,p.z, 
		      l.x,l.y,l.z,
			  u.x,u.y,u.z);

	frustum.setCamDef(p,l,u);

	render();

	frame++;
	time=glutGet(GLUT_ELAPSED_TIME);
	if (time - timebase > 1000) {
		fps = frame*1000.0/(time-timebase);
		timebase = time;
		frame = 0;
		sprintf(title, "Frustum:%d  Spheres Drawn: %d Total Spheres: %d FPS: %8.2f",
					frustumOn,spheresDrawn,spheresTotal,fps);
		glutSetWindowTitle(title);
	}

	glutSwapBuffers();
}




void keyboard(unsigned char a, int x, int y) {

	Vec3 v;

	switch(a) {

		case 'w': 
		case 'W': 
			v.set(0,1,0);
			p = p + v*0.1;
			l = l + v*0.1;
			break;

		case 's': 
		case 'S': 
			v.set(0,1,0);
			p = p - v*0.1;
			l = l - v*0.1;
			break;
		case 'd': 
		case 'D': 
			v.set(1,0,0);
			p = p + v*0.1;
			l = l + v*0.1;
			break;

		case 'a': 
		case 'A': 
			v.set(0,1,0);
			p = p - v*0.1;
			l = l - v*0.1;
			break;

		case 't':
		case 'T':
			v.set(0,0,1);
			p = p - v * 0.1;
			l = l - v * 0.1;
			break;
		case 'g':
		case 'G':
			v.set(0,0,1);
			p = p + v * 0.1;
			l = l + v * 0.1;
			break;

		case 'r':
		case 'R':
			p.set(0,0,5);
			l.set(0,0,0);
			u.set(0,1,0);
			break;

		case 'f':
		case 'F':
			frustumOn = !frustumOn;
			break;
		case 27: 
			exit(0);
			break;


	}
}

void main(int argc, char **argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,640);
	glutCreateWindow("Camera View");

	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(keyboard);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	printf("Help Screen\n\nW,S,A,D: Pan vertically and Horizontally\nT,G:Move forward/backward\nR: Reset Position\n\nF: Turn frustum On/Off");

	glutMainLoop();
}

