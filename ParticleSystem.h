#pragma once
#include <typeinfo>
#define MAX_PARTICLES 10000
#define GRAVITY 0.32f
#define YSPEED 0.90f
#define SPEED 0.1f
#define SIZESCALE 0.05f

#ifndef BASICS
#define BASICS

#include "CShader.h"
#include "CMeshLoader.h"
#include "Util/SSphere.h"
#include "BVHNode.h"

#endif

typedef struct 
{
  SSphere sphere;
  SVector3 velocity;
} Particle;

class ParticleSystem {

public:
   
   CShader* shade;
   GLuint PositionBufferHandle, ColorBufferHandle, NormalBufferHandle;
   SVector3 Translation, Rotation, Scale;
   int TriangleCount;
   float size, random, bounce, speed;
   int numParticles;
   Particle particles[MAX_PARTICLES];

   ParticleSystem(SVector3 pos, float random, CMesh * mod, float size, float bounce, float speed);
   ~ParticleSystem();
   void update(float dt);
   void draw();
   void setNumParticles(int num);
   int getNumParticles();
   void moveParticle(int ndx, float dt);
   void resetParticles();
   void resetParticle(int i);
   int checkTriangle(SVector3 A, SVector3 B, SVector3 C, SVector3 center, float radius, SVector3 vel);
   void collideWith(std::vector<SSphere> spheres);
   void collideWithBVH(BVHNode* head);
};


extern "C" void cudaUpdate(ParticleSystem *psys, float time);
extern "C" void CUDAcollideWithBVH(ParticleSystem *psys, CUDA_BVH* head);
extern "C" void CUDAcollideWith(ParticleSystem *psys, std::vector<SSphere> spheres);
