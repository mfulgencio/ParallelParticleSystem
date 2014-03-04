
#include <GL/glut.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "GameObject.h"

GameObject::GameObject(SVector3* pos, SVector3* vel, CMesh * mod)
{
   position = pos;
   velocity = vel;
   model = mod;
}

GameObject::~GameObject()
{
}

void draw() 
{
}
void GameObject::collideWith(GameObject* collided)
{
}

void GameObject::update(float dt)
{
   this->position->X += this->velocity->X * dt;
   this->position->Y += this->velocity->Y * dt;
   this->position->Z += this->velocity->Z * dt;
}

SVector3* GameObject::getPosition()
{
   return position;
}
