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


#define GRAVITY 0.008f
#define YSPEED 0.01f
#define SPEED 0.002f


ParticleSystem::ParticleSystem(SVector3* pos, float random, CMesh* mod, float size) {
  this->size = size;
  this->random = random;
  this->numParticles = 200;

  Translation.X = pos->X;
  Translation.Y = pos->Y;
  Translation.Z = pos->Z;

  Scale.X = size; 
  Scale.Y = size;
  Scale.Z = size;

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
	shade = ShaderLoader.loadShader("Shaders/GameVert2.glsl", "Shaders/Lab3_frag.glsl");
	if (! shade)
	{
		std::cerr << "Unable to open or compile necessary shader." << std::endl;
	}
	shade->loadAttribute("aPosition");
	shade->loadAttribute("aColor");
  shade->loadAttribute("aNormal");
	
	// Attempt to load mesh
  mod = CMeshLoader::loadASCIIMesh("Models/bullet.obj"); /** TODO change this for the love of god **/
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
    particles[i].offset = new SVector3();
    particles[i].velocity = new SVector3();
  }
  resetParticles();
}

ParticleSystem::~ParticleSystem() { }

void ParticleSystem::moveParticle(int i, float time)
{
   particles[i].velocity->Y -= GRAVITY * time;

   particles[i].offset->X += particles[i].velocity->X; 
   particles[i].offset->Y += particles[i].velocity->Y; 
   particles[i].offset->Z += particles[i].velocity->Z; 
}


void ParticleSystem::update(float time)
{
  for (int i = 0; i < numParticles; i++)
	{
     moveParticle(i, time);
     if (particles[i].offset->Y < -2)
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

		glTranslatef(particles[i].offset->X, particles[i].offset->Y, particles[i].offset->Z);
		glRotatef(Rotation.X, 1, 0, 0);
		glRotatef(Rotation.Y, 0, 1, 0);
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
  particles[i].offset->X = Translation.X + rbFloat() * random;
  particles[i].offset->Y = Translation.Y + rbFloat() * random;
  particles[i].offset->Z = Translation.Z + rbFloat() * random;
  
  particles[i].velocity->X = rbFloat() * SPEED;
  particles[i].velocity->Y = -rFloat() * YSPEED;
  particles[i].velocity->Z = rbFloat() * SPEED;
  
}

