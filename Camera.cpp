#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "Camera.h"

#define DEGTORAD 3.1415926 / 180
#define MOVEMENTSPEED 1.8

Camera::Camera (int x, int y, int z, Player* p)
{
	Position.X = x;
	Position.Y = y;
	Position.Z = z;

	Direction.X = 0;
	Direction.Y = 0;
	Direction.Z = 1;

	player = p;
}

Camera::~Camera(){
	
}

void Camera::update()
{
   /** position is updated from InputManager **/

   Direction = *player->getPosition() - Position + SVector3(0,0,0);
   Direction /= Direction.length(); // normalize
}

void Camera::setLookAt()
{
   gluLookAt(
      Position.X, Position.Y + 1, Position.Z, 
      0, 0, 0, 
      0, 1, 0);
}

SVector3 Camera::getPosition()
{
   return Position;
}
