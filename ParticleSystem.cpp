#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <iostream>

#include <stdio.h>
#include "ParticleSystem.h"
#include <stdlib.h>


#define GRAVITY 0.32f
#define YSPEED 0.90f
#define SPEED 0.1f
#define SIZESCALE 0.05f

ParticleSystem::ParticleSystem(SVector3 pos, float random, CMesh* mod, float size, float bounce, float speed) {
  this->size = size * SIZESCALE;
  this->bounce = bounce;
  this->speed = speed;
  this->random = random;
  this->numParticles = 1000;

  Translation = pos;

  Scale.X = this->size; 
  Scale.Y = this->size;
  Scale.Z = this->size;

  Rotation.X = 0;
  Rotation.Y = 0;
  Rotation.Z = 0;

  // First create a shader loader and check if our hardware supports shaders
	CShaderLoader ShaderLoader;
	if (! ShaderLoader.isValid())
	{
		std::cerr << "Shaders are not supported by your graphics hardware, or the shader loader was otherwise unable to load." << std::endl;
	}

	// Now attempt to load the shaders
	shade = ShaderLoader.loadShader("Shaders/GameVert1.glsl", "Shaders/Lab3_frag.glsl");
	if (! shade)
	{
		std::cerr << "Unable to open or compile necessary shader." << std::endl;
	}
	shade->loadAttribute("aPosition");
	shade->loadAttribute("aColor");
  shade->loadAttribute("aNormal");
	// Attempt to load mesh
  mod = CMeshLoader::loadASCIIMesh("Models/sphere.m"); 
	if (! mod)
	{
		std::cerr << "Unable to load necessary mesh." << std::endl;
	}

	// Make out mesh fit within camera view
	mod->resizeMesh(SVector3(1));
	// And center it at the origin
	mod->centerMeshByExtents(SVector3(0));

	// Now load our mesh into a VBO, retrieving the number of triangles and the handles to each VBO
	CMeshLoader::createVertexBufferObject(* mod, TriangleCount, 
		PositionBufferHandle, ColorBufferHandle, NormalBufferHandle);

  for (int i = 0; i < MAX_PARTICLES; i++)
  {
    particles[i].sphere = SSphere(SVector3(), this->size);
    particles[i].velocity = SVector3();
  }
  resetParticles();
}

ParticleSystem::~ParticleSystem() { }

void ParticleSystem::moveParticle(int i, float time)
{
   particles[i].velocity.Y -= GRAVITY * time;

   particles[i].sphere.center.X += particles[i].velocity.X * time * this->speed; 
   particles[i].sphere.center.Y += particles[i].velocity.Y * time * this->speed; 
   particles[i].sphere.center.Z += particles[i].velocity.Z * time * this->speed; 
}


void ParticleSystem::update(float time)
{
  for (int i = 0; i < numParticles; i++)
	{
     moveParticle(i, time);
     if (particles[i].sphere.center.Y < -2)
        resetParticle(i);
  }
}

void ParticleSystem::draw()
{
  for (int i = 0; i < numParticles; i++)
	{
		// Shader context works by cleaning up the shader settings once it
		// goes out of scope
		CShaderContext ShaderContext(*shade);
		ShaderContext.bindBuffer("aPosition", PositionBufferHandle, 4);
		ShaderContext.bindBuffer("aColor", ColorBufferHandle, 3);
		ShaderContext.bindBuffer("aNormal", NormalBufferHandle, 3);

		glPushMatrix();

		glTranslatef(particles[i].sphere.center.X, particles[i].sphere.center.Y, particles[i].sphere.center.Z);
    if (1) // should they randomly rotate?
    {
		  glRotatef(i, 1, 0, 0);
		  glRotatef((i * i * i), 0, 1, 0);
    }
		glScalef(Scale.X, Scale.Y, Scale.Z);

		glDrawArrays(GL_TRIANGLES, 0, TriangleCount*3);

		glPopMatrix();
	}
}

int ParticleSystem::getNumParticles() { return numParticles; }
void ParticleSystem::setNumParticles(int num) { 
  if (num > MAX_PARTICLES)
  {
    printf("max particles reached!\n");
    return;
  }
  if (num < 10)
  {
    printf("min particles reached!\n");
    return;
  }

  numParticles = num; 
  resetParticles();
}

float rFloat ()
{
  return ((float)rand()) / RAND_MAX;
}

float rbFloat ()
{
  return rFloat() * 2 - 1;
}

void ParticleSystem::resetParticles()
{
  for (int i = 0; i < numParticles; i++)
  {
    resetParticle(i);
  }
}


void ParticleSystem::resetParticle(int i)
{
  particles[i].sphere.center.X = Translation.X + rbFloat() * random;
  particles[i].sphere.center.Y = Translation.Y + rbFloat() * random;
  particles[i].sphere.center.Z = Translation.Z + rbFloat() * random;
  
  particles[i].velocity.X = rbFloat() * SPEED;
  particles[i].velocity.Y = -rFloat() * YSPEED;
  particles[i].velocity.Z = rbFloat() * SPEED;
}


int ParticleSystem::checkTriangle(SVector3 A, SVector3 B, SVector3 C, SVector3 center, float radius, SVector3 vel)
{
  //return 1;

  SVector3 normal = ((A - C).crossProduct(A - B));
  normal /= normal.length();

  SVector3 dirA = (A - (center - (normal * -radius)));
  SVector3 dirB = (A - (center + (normal * -radius)));

  float dot1 = dirA.dotProduct(normal);
  float dot2 = dirB.dotProduct(normal);

  if (dot1 > 0 && dot2 > 0 || dot1 < 0 && dot2 < 0)
    return 0;
  return 1;
}

void ParticleSystem::collideWith(std::vector<SSphere> spheres)
{
  for (int i = 0; i < numParticles; i++)
  {
    for (int j = 0; j < spheres.size(); j++)
    {
      if (spheres[j].collidesWith(particles[i].sphere) && 
          checkTriangle(spheres[j].A, spheres[j].B, spheres[j].C, particles[i].sphere.center, particles[i].sphere.radius, particles[i].velocity))
      {
        float len = particles[i].velocity.length();
        SVector3 dir = (particles[i].sphere.center) - spheres[j].center; 
        dir /= dir.length();
        dir *= len * this->bounce;        

        particles[i].velocity = dir;
        particles[i].sphere.center += (particles[i].velocity) * this->size;

        break;
      }
    }
  }
}

void ParticleSystem::collideWithBVH(BVHNode* head)
{
  for (int i = 0; i < numParticles; i++)
  {
    SSphere* hit = head->checkHit(particles[i].sphere);
    if (hit != NULL && !hit->isEmpty() && checkTriangle(hit->A, hit->B, hit->C, particles[i].sphere.center, particles[i].sphere.radius, particles[i].velocity))
    {
      float len = particles[i].velocity.length();
      SVector3 dir = (particles[i].sphere.center) - hit->center; 
      dir /= dir.length();
      dir *= len * this->bounce;        

      particles[i].velocity = dir;
      particles[i].sphere.center += (particles[i].velocity) * this->size;
    }
  }
}
