#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif

#ifdef __unix__
#include <iostream>
#include <GL/glut.h>
#include <stdio.h>
#include <time.h>
#endif

// Utility classes for loading shaders/meshes
#include "CShader.h"
#include "CMeshLoader.h"
#include "Camera.h"
#include "HUD.h"
#include "InputManager.h"
#include "Player.h"
#include "ParticleSystem.h"
#include "GameObject.h"
#include "Util/Vector.hpp"
#include <stdlib.h>
#include <string.h>


/********************
 * Global Varaibles *
 ********************/


// Window information
int WindowWidth = 1000, WindowHeight = 600;

// Time-independant movement variables
int Time0, Time1;

int prevX, prevY;
int w = 0, a = 0, s = 0, d = 0;

int curTime = 0;
int numFrames = 0, lastSecond = 0, FPS = 0;
time_t *timeTracker;

Camera *camera;
Player *player;
ParticleSystem *psys;
HUD* hud;
InputManager* manager;


/***************************
 * GLUT Callback Functions *
 ***************************/

// OpenGL initialization
void Initialize()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
 	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// Start keeping track of time
	Time0 = glutGet(GLUT_ELAPSED_TIME);
}


void update(float dtime)
{
  player->update(dtime);
  psys->update(dtime);
  psys->collideWith(player->hitspheres);
  manager->update();
  camera->update();
}
   
// Manages time independent movement and draws the VBO
void Display()
{
	// Determine time since last draw
	Time1 = glutGet(GLUT_ELAPSED_TIME);
	float Delta = (float) (Time1 - Time0) / 1000.f;
	Time0 = Time1;

	update(Delta);

  long dtime = (time(NULL) - *timeTracker) * 1000; 
	curTime += dtime;
	numFrames++;
  if (curTime / 1000 > lastSecond)
	{
	  lastSecond++;
	  FPS = numFrames;
	  numFrames = 0;
	}
	time(timeTracker);
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
  
  camera->setLookAt();
  player->draw();
  psys->draw();
  hud->drawText(FPS, curTime, psys->getNumParticles());

	glutSwapBuffers();
	glutPostRedisplay();
}

void Reshape(int width, int height)								
{
	glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));
	WindowWidth = width;
	WindowHeight = height;

	// Set camera projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float AspectRatio = (float)WindowWidth / (float)WindowHeight;
	gluPerspective(60.0, AspectRatio, 0.01, 1000.0);
}

void keyCallback(unsigned char key, int x, int y) {
  manager->keyCallBack(key, x, y);
}

void keyUpCallback(unsigned char key, int x, int y) {
  manager->keyUpCallBack(key, x, y);
}

void mouseMotion(int x, int y)
{
	manager->mouseMotion(x,y);
}


const char* getModelName(int argc, char* argv[])
{
  for (int i = 0; i < argc - 1; i++)
  {
    if (strcmp(argv[i], "-model") == 0)
      return argv[i + 1];
  }
  return "Models/bunny500.m";
}

float getFloat(int argc, char* argv[], const char* name, float def)
{
  for (int i = 0; i < argc - 1; i++)
  {
    if (strcmp(argv[i], name) == 0)
      return strtof(argv[i + 1], NULL);
  }
  return def;
}

int main(int argc, char* argv[])
{
	glutInit(& argc, argv);
 	glutInitWindowPosition(100, 200);
 	glutInitWindowSize(WindowWidth, WindowHeight);
 	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

 	glutCreateWindow("CPE 419 - Parallelized Particle Collision Detection");
	glutReshapeFunc(Reshape);
 	glutDisplayFunc(Display);
	
  timeTracker = (time_t*)malloc(sizeof(time_t));
  *timeTracker = time(timeTracker);

	prevX = 200;
	prevY = 200;
	glutSetCursor(GLUT_CURSOR_NONE); 
	
  printf("Loading data\n");

 	Initialize();
  float size = 1.0;

	player = new Player(new SVector3(0,0,0), NULL, size, getModelName(argc, argv));
  psys = new ParticleSystem(SVector3(0,2,0), getFloat(argc, argv, "-random", 0.5f), NULL, 
                            getFloat(argc, argv, "-size", 0.05f), getFloat(argc, argv, "-bounce", 0.8f), 
                            getFloat(argc, argv, "-speed", 1));
  if (getFloat(argc, argv, "-n", 0) != 0)
  {
    psys->setNumParticles((int)getFloat(argc, argv, "-n", 200));
  } 
	camera = new Camera(0, 0, -3, player);
  manager = new InputManager(player, camera, psys);
	hud = new HUD();

  manager->sendPlayerPositionPulse();  
   
  printf("Opening Window\n"); 

	glutKeyboardFunc(keyCallback);
	glutKeyboardUpFunc(keyUpCallback);
 	glutPassiveMotionFunc(mouseMotion);
	// ... and begin!
	glutMainLoop();
	
	return 0;
}
