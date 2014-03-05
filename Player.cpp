#include <iostream>
#include <GL/glut.h>

#include <stdio.h>
#include "Player.h"

void waitForUser3() 
{
	std::cout << "Press [Enter] to continue . . .";
	std::cin.get();
}

Player::Player(SVector3* pos, CMesh* mod, float size, const char* name) 
   : GameObject(pos, new SVector3(0,0,0), mod) {
  this->size = size;

  Translation.X = pos->X;
  Translation.Y = pos->Y;
  Translation.Z = pos->Z;

  Scale.X = 1; 
  Scale.Y = 1;
  Scale.Z = 1;

  Rotation.X = 0;
  Rotation.Y = 0;
  Rotation.Z = 0;

  // First create a shader loader and check if our hardware supports shaders
	CShaderLoader ShaderLoader;
	if (! ShaderLoader.isValid())
	{
		std::cerr << "Shaders are not supported by your graphics hardware, or the shader loader was otherwise unable to load." << std::endl;
		waitForUser3();
	}

	// Now attempt to load the shaders
	shade = ShaderLoader.loadShader("Shaders/GameVert2.glsl", "Shaders/Lab3_frag.glsl");
	if (! shade)
	{
		std::cerr << "Unable to open or compile necessary shader." << std::endl;
		waitForUser3();
	}
	shade->loadAttribute("aPosition");
	shade->loadAttribute("aColor");
  shade->loadAttribute("aNormal");
	
	// Attempt to load mesh
  mod = CMeshLoader::loadASCIIMesh(name);
	if (! mod)
	{
		std::cerr << "Unable to load necessary mesh." << std::endl;
		waitForUser3();
	}
	// Make out mesh fit within camera view
	mod->resizeMesh(SVector3(1));
	// And center it at the origin
	mod->centerMeshByExtents(SVector3(0));

  this->hitspheres = mod->gimmeSpheres();

	// Now load our mesh into a VBO, retrieving the number of triangles and the handles to each VBO
	CMeshLoader::createVertexBufferObject(* mod, TriangleCount, 
		PositionBufferHandle, ColorBufferHandle, NormalBufferHandle);

}

Player::~Player()
{ }

SVector3* Player::getPosition()
{
   return position;
}SVector3* Player::getVelocity()
{
   return velocity;
}
SVector3* Player::getTranslation()
{
   return &Translation;
}

void Player::update(float dt)
{
	Translation.X = position->X;
	Translation.Y = position->Y;
	Translation.Z = position->Z;
}
float Player::getSize()
{
   return 1;
}
void Player::draw()
{
	{
		// Shader context works by cleaning up the shader settings once it
		// goes out of scope
		CShaderContext ShaderContext(*shade);
		ShaderContext.bindBuffer("aPosition", PositionBufferHandle, 4);
		ShaderContext.bindBuffer("aColor", ColorBufferHandle, 3);
		ShaderContext.bindBuffer("aNormal", NormalBufferHandle, 3);

		glPushMatrix();

		glTranslatef(Translation.X, Translation.Y, Translation.Z);
		glRotatef(Rotation.X, 1, 0, 0);
		glRotatef(Rotation.Y, 0, 1, 0);
		glScalef(Scale.X, Scale.Y, Scale.Z);

		glDrawArrays(GL_TRIANGLES, 0, TriangleCount*3);

		glPopMatrix();
	}
}
void Player::collideWith(GameObject* collided)
{
  /** TODO if particles are GameObjects, put collision detection here? **/
}




