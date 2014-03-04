#pragma once
#include <typeinfo>
#define MAX_PARTICLES 10000

#ifndef BASICS
#define BASICS

#include "CShader.h"
#include "CMeshLoader.h"

#endif

typedef struct 
{
  SVector3* offset;
  SVector3* velocity;
} Particle;

class ParticleSystem {

public:
   
   CShader* shade;
   GLuint PositionBufferHandle, ColorBufferHandle, NormalBufferHandle;
   SVector3 Translation, Rotation, Scale;
   int TriangleCount;
   float size, random;
   int numParticles;
   Particle particles[MAX_PARTICLES];

   ParticleSystem(SVector3* pos, float random, CMesh * mod, float size);
   ~ParticleSystem();
   void update(float dt);
   void draw();
   void setNumParticles(int num);
   int getNumParticles();
   void moveParticle(int ndx, float dt);
   void resetParticles();
   void resetParticle(int i);

};
