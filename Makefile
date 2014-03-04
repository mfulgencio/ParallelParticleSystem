all:
	g++ -o test MainFull.cpp CMeshLoader.cpp CShader.cpp CMesh.cpp GameObject.cpp InputManager.cpp Player.cpp Camera.cpp HUD.cpp MeshParser.cpp ParticleSystem.cpp Util/SSphere.cpp  -DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU

run: 
	./test

do: all run
