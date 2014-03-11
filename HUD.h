#pragma once

class HUD 
{
public:
	HUD();
	~HUD();
	void renderBitmapString (float x, float y, float z, char *string);
	void renderGlutAimer(float px, float py, float dx, float dy);
	void drawText(int fps, int curTime, int parts, int* FPSs);
	void drawFPS(int* FPSs, int num, const char* str, float x, float y);
        void drawWin();
        void drawLose();
};
