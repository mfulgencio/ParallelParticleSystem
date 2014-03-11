#pragma once
#include "GameObject.h"
#include "BVHNode.h"

#include <typeinfo>

#ifndef BASICS
#define BASICS

#include "CShader.h"
#include "CMeshLoader.h"

#endif

class Player : GameObject
{
public:

   float size;
   CShader* shade;

   // Handles for VBOs
   GLuint PositionBufferHandle, ColorBufferHandle, NormalBufferHandle;

   // Information about mesh
   SVector3 Translation, Rotation, Scale;
   int TriangleCount;
   std::vector<SSphere> hitspheres;
   BVHNode *head;

   Player(SVector3* pos, CMesh * mod, float size, const char* name);
   ~Player();
   void update(float dt);
   void draw();
   float getSize();
   void collideWith(GameObject* collided);
   SVector3* getPosition();
   SVector3* getTranslation();
   SVector3* getVelocity();
   BVHNode* constructBVH(int startNdx, int endNdx);
};
