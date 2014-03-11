#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#include <stdlib.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include "InputManager.h"
#include "Player.h"
#include "Camera.h"
#include <stdio.h>
#include <math.h>

InputManager::InputManager(Player* pl, Camera* c, ParticleSystem *psys)
{
   this->player = pl;
   this->camera = c;
   this->psys = psys;
   SENSITIVITY = 80;
   AbsX = 0;
   AbsY = 0;
   dx = 0;
   dy = 0;
   prevX = 0;
   prevY = 0;
   glutWarpPointer(500, 300);
   radius = 3.5f;
   theta = 0;

   a = d = s = w = o = p = r = f = 0;
}

InputManager::~InputManager(){
}

void InputManager::keyCallBack(unsigned char key, int x, int y) {
   
   if (key == 'w') w = 1;
   if (key == 's') s = 1;
   if (key == 'a') a = 1;
   if (key == 'd') d = 1;
   if (key == 'o') o = 1;
   if (key == 'p') p = 1;
   if (key == 'r') r = 1;
   if (key == 'f') f = 1;
   
}

void InputManager::keyUpCallBack(unsigned char key, int x, int y) {
   if (key == 27)
	   exit(0);

   if (key == 'w') w = 0;
   if (key == 's') s = 0;
   if (key == 'a') a = 0;
   if (key == 'd') d = 0;
   if (key == 'o') o = 0;
   if (key == 'p') p = 0;
   if (key == 'r') r = 0;
   if (key == 'f') f = 0;

}

void InputManager::update()
{
   float speed = 0.05f;
   if (a) {
      this->camera->Position.X += speed;
  //    this->camera->Position.Y += speed;
   }
   if (d) {
      this->camera->Position.X -= speed;
    //  this->camera->Position.Y -= speed;
   }
   if (w) {
      this->camera->Position.Y += speed;
   }
   if (s) {
      this->camera->Position.Y -= speed;
   }
   if (a) theta += speed;
   if (d) theta -= speed;
   this->camera->Position.X = cos(theta) * radius; 
   this->camera->Position.Z = sin(theta) * radius; 
   if (w) this->camera->Position.Y += speed;
   if (s) this->camera->Position.Y -= speed;
   if (r) radius -= speed;
   if (f) radius += speed;

   if (radius < 2) radius = 2;

   if (o) this->psys->setNumParticles(this->psys->getNumParticles() + 10);
   if (p) this->psys->setNumParticles(this->psys->getNumParticles() - 10);
}

void InputManager::sendPlayerPositionPulse()
{
   glutWarpPointer(500, 300);
}
void InputManager::mouseMotion(int x, int y) {
   dx = prevX - x;
   dy = prevY - y;
   prevX = x;
   prevY = y;
   // assuming 500,500 is the glutWarpedPointer location
   AbsX = (x - 500) / SENSITIVITY;
   AbsY = (300 - y) / SENSITIVITY;
}
