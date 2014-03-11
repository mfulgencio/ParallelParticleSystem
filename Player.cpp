#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <iostream>

#include <stdio.h>
#include "Player.h"


void waitForUser3() 
{
	std::cout << "Press [Enter] to continue . . .";
	std::cin.get();
}
int indexOfSmallest(int start, int end, std::vector<SSphere> *spheres, char dim)
{
  int min = start;
  for (int i = start + 1; i < end; i++)
  {
    switch(dim) {
      case 'x':
        if ((*spheres)[i].center.X < (*spheres)[min].center.X) min = i; break;
      case 'y':
        if ((*spheres)[i].center.Y < (*spheres)[min].center.Y) min = i; break;
      case 'z':
        if ((*spheres)[i].center.Z < (*spheres)[min].center.Z) min = i; break;
    }
  }
  return min;
}
void swap(std::vector<SSphere> *spheres, int i, int j)
{
   float tx = (*spheres)[i].center.X, ty = (*spheres)[i].center.Y, tz = (*spheres)[i].center.Y, tr = (*spheres)[i].radius;

   (*spheres)[i].center.X = (*spheres)[j].center.X;
   (*spheres)[i].center.Y = (*spheres)[j].center.Y;
   (*spheres)[i].center.Z = (*spheres)[j].center.Z;
   (*spheres)[i].radius = (*spheres)[j].radius;

   (*spheres)[j].center.X = tx;
   (*spheres)[j].center.Y = ty;
   (*spheres)[j].center.Z = tz;
   (*spheres)[j].radius = tr;
}
void selectionSort(std::vector<SSphere> *spheres)
{
  // first pass, sort along x axis
  for (int i = 0; i < spheres->size() - 1; i++)
  {
    int ndx = indexOfSmallest(i, spheres->size(), spheres, 'x');
    swap(spheres, i, ndx);
  }

  // second pass, very coarsely sort along y axis
  for (int i = 0; i < spheres->size() - spheres->size() % 32; i+= 32)
  {
    for (int j = 0; j < 32; j++)
    {
      int ndx = indexOfSmallest(i + j, i + 32, spheres, 'y');
      swap(spheres, i, ndx);
    }
  }

  // third pass, coarsely sort along z axis
  for (int i = 0; i < spheres->size() - spheres->size() % 8; i+= 8)
  {
    for (int j = 0; j < 8; j++)
    {
      int ndx = indexOfSmallest(i + j, i + 8, spheres, 'z');
      swap(spheres, i, ndx);
    }
  }
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
  printf("Creating BVH\n");
  //selectionSort(&(this->hitspheres));
  this->head = constructBVH(0, this->hitspheres.size() - 1);
  this->head->createCudaBVH(this->bvh, 0);

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

BVHNode* Player::constructBVH(int startNdx, int endNdx)
{
   if (startNdx == endNdx)
   {
      return new BVHNode(this->hitspheres[startNdx]);
   }
   else
   {
      int mid = startNdx + (endNdx - startNdx) / 2;
      return new BVHNode(constructBVH(startNdx, mid), constructBVH(mid + 1, endNdx));
   }
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
		//glRotatef(Rotation.Z, 0, 0, 1);
		glScalef(Scale.X, Scale.Y, Scale.Z);

		glDrawArrays(GL_TRIANGLES, 0, TriangleCount*3);

		glPopMatrix();
	}
}
void Player::collideWith(GameObject* collided)
{
}




