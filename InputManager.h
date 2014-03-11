#pragma once
#include "Player.h"
#include "Camera.h"
#include "ParticleSystem.h"

class InputManager{
public:
float SENSITIVITY;

int w, a, s, d, o, p, r, f, dx, dy, prevX, prevY;
float AbsX, AbsY, radius, theta;
Player* player;
Camera* camera;
ParticleSystem* psys;

InputManager(Player* p, Camera* c, ParticleSystem *psys);
~InputManager();
void sendPlayerPositionPulse();
void keyCallBack(unsigned char key, int x, int y);
void keyUpCallBack(unsigned char key, int x, int y);
void mouseMotion(int x, int y);
void update();
};
