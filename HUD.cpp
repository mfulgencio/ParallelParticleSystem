#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <iostream>

#include <stdio.h>
#include "HUD.h"
#include <string.h>

HUD::HUD()
{ }

HUD::~HUD()
{ }

void HUD::renderBitmapString (float x, float y, float z, char *string) 
{  
  char *c;
  glRasterPos3f(x, y, z);
  for (c=string; *c != '\0'; c++) 
  {
     glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
  }
}

int av(int* arr, int len)
{
  float ret = 0;
  for (int i = 0; i < len; i++) {
     ret += arr[i];
     if (arr[i] == -1) return 0; 
  }
  ret /= len;
  return ret;
}

void HUD::drawFPS(int* FPSs, int num, const char* str, float x, float y)
{
	char fps[20];
  for (int i = 0; i < 20; i++) fps[i] = 0;
  strcpy(fps, str);
  int FPS = av(FPSs, num);
	fps[strlen(str)] = FPS/100 % 10 + 48;
	fps[strlen(str) + 1] = FPS/10 % 10 + 48;
	fps[strlen(str) + 2] = FPS/1 % 10 + 48;
	renderBitmapString(x, y, -4.5, fps);
}

void HUD::drawText(int FPS, int curTime, int numParts, int* FPSs)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
 	gluLookAt(0.0, 0.0, 1.0, 
            0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0);
	GLfloat color[3] = {0.4, 1.0, 0.1};
	
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color); 
		
 	GLfloat lightDir[] = {0, 0, -1, 0.0};
	GLfloat diffuseComp[] = {1.0, 1.0, 1.0, 1.0};

	glEnable(GL_LIGHT0);
 	glLightfv(GL_LIGHT0, GL_POSITION, lightDir);
 	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseComp);
   
  glDisable(GL_DEPTH);
  glDisable(GL_DEPTH_TEST);

	char timer[14];
	timer[0] = (((int)curTime)/100000) % 10 + 48;
	timer[1] = (((int)curTime)/10000) % 10 + 48;
	timer[2] = (((int)curTime)/1000) % 10 + 48;
	timer[3] ='s';
	timer[4]='\0';
	renderBitmapString(-0.5, 2.7, -4.5, timer);

	char fps[9] = "FPS: ";
	fps[5] = FPS/100 % 10 + 48;
	fps[6] = FPS/10 % 10 + 48;
	fps[7] = FPS/1 % 10 + 48;
	renderBitmapString(-0.75, 2.5, -4.5, fps);

	char parts[7] = "N: ";
	parts[3] = numParts/1000 % 10 + 48;
	parts[4] = numParts/100 % 10 + 48;
	parts[5] = numParts/10 % 10 + 48;
	parts[6] = numParts/1 % 10 + 48;
	parts[7]='\0';
	renderBitmapString(-0.7, 2.3, -4.5, parts);

  drawFPS(FPSs, 10, "FPS (10 sec): ", 0.7, 2.7);
  drawFPS(FPSs, 30, "FPS (30 sec): ", 0.7, 2.5);
  drawFPS(FPSs, 60, "FPS (60 sec): ", 0.7, 2.3);

	glDisable(GL_LIGHT0);
   
  glEnable(GL_DEPTH);
  glEnable(GL_DEPTH_TEST);
}


void HUD::renderGlutAimer(float px, float py, float dx, float dy)
{
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   gluLookAt(0.0, 0.0, 1.0, 
            0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0);
   GLfloat color[3] = {0.4, 1.0, 0.1};
	
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color); 
		
   GLfloat lightDir[] = {0, -1, -1, 0.0};
   GLfloat diffuseComp[] = {0.1, 0.1, 0.1, 1.0};

   glEnable(GL_LIGHT0);
   glLightfv(GL_LIGHT0, GL_POSITION, lightDir);
   glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseComp);

   glTranslatef(dx, dy, -9);
   glutSolidTorus(0.02, 0.6, 8, 8);
   glTranslatef(-dx, -dy, 0);
   dx = 0.4 * px + 0.6 * dx;
   dy = 0.4 * py + 0.6 * dy;

   glTranslatef(dx, dy, 2);
   glutSolidTorus(0.02, 0.7, 8, 8);
   glTranslatef(-dx, -dy, 0);
   dx = 0.5 * px + 0.5 * dx;
   dy = 0.5 * py + 0.5 * dy;

   glTranslatef(dx, dy, 2);
   glutSolidTorus(0.02, 0.8, 8, 8);
   glTranslatef(-dx, -dy, 0);

   glDisable(GL_LIGHT0);
}


