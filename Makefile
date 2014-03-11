all:
	nvcc -c cudaFunctions.cu
	nvcc -o run MainFull.cpp BVHNode.cpp CMeshLoader.cpp CShader.cpp CMesh.cpp GameObject.cpp InputManager.cpp Player.cpp Camera.cpp HUD.cpp MeshParser.cpp ParticleSystem.cpp Util/SSphere.cpp cudaFunctions.o -DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU


apple:
	g++ -o run MainFull.cpp BVHNode.cpp CMeshLoader.cpp CShader.cpp CMesh.cpp GameObject.cpp InputManager.cpp Player.cpp Camera.cpp HUD.cpp MeshParser.cpp ParticleSystem.cpp Util/SSphere.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT

run: all
	./run
