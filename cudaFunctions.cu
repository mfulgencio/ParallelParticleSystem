/* cudaFunctions.cu
 *
 * Contains the code for all the cuda functions used
 *
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "ParticleSystem.h"
#include "Util/SVector3.h"
#include "Util/SSphere.h"

#define THREADS_PER_BLOCK 32

// Variables needed for moveParticle(): ParticleSystem.particles[], ParticleSystem.speed,
   // particle index, float "time" variable, GRAVITY constant
// Variables needed for resetParticle(): ParticleSystem.particles[], particle index,
   // SVector3 translation, rbFloat() & rFloat(), SPEED & YSPEED constants
   
typedef struct {
   Particle particles[MAX_PARTICLES];
   int numParticles;
   float speed;
   float random;
   SVector3 Translation;
} CudaParticleSystem;


__device__ float rFloat(curandState *randStates)  {
   int id = (blockIdx.x * blockDim.x) + threadIdx.x;
   curandState localState = randStates[id];
   float randNum = (float)curand_uniform(&localState);
   randStates[id] = localState;
   
   return randNum;
}


__device__ float rbFloat(curandState *randStates) {
   return rFloat(randStates) * 2 - 1;
}


__device__ void moveParticle(Particle particle, float speed, float time) {
   particle.velocity.Y -= GRAVITY * time;

   particle.sphere.center.X += particle.velocity.X * time * speed; 
   particle.sphere.center.Y += particle.velocity.Y * time * speed; 
   particle.sphere.center.Z += particle.velocity.Z * time * speed; 
}


__device__ void resetParticle(curandState *randStates, Particle particle, SVector3 Translation, float random, float time) {
   particle.sphere.center.X = Translation.X + rbFloat(randStates) * random;
   particle.sphere.center.Y = Translation.Y + rbFloat(randStates) * random;
   particle.sphere.center.Z = Translation.Z + rbFloat(randStates) * random;
   
   particle.velocity.X = rbFloat(randStates) * SPEED;
   particle.velocity.Y = -rFloat(randStates) * YSPEED;
   particle.velocity.Z = rbFloat(randStates) * SPEED;
}


__global__ void update(curandState *randStates, CudaParticleSystem *cpsys, float time) {
   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
   Particle curParticle = cpsys->particles[index];
   
   moveParticle(curParticle, cpsys->speed, time);
   
   if(curParticle.sphere.center.Y < -2) {
      curand_init(1234, index, 0, &randStates[index]);
      resetParticle(randStates, curParticle, cpsys->Translation, cpsys->random, time);
   }
}


extern "C" void cudaUpdate(ParticleSystem *psys, float time) {
   int num_blocks = 0;
   curandState *randStates;

   CudaParticleSystem *cpsys_device;
   
   cudaMalloc((void **)&cpsys_device, sizeof(CudaParticleSystem));
   cudaMalloc((void **)&randStates, MAX_PARTICLES * sizeof(curandState));
   
   cudaMemcpy(cpsys_device->particles, psys->particles, sizeof(Particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->numParticles, &psys->numParticles, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->speed, &psys->speed, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->random, &psys->random, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->Translation, &psys->Translation, sizeof(SVector3), cudaMemcpyHostToDevice);
   
   num_blocks = ceil(MAX_PARTICLES / THREADS_PER_BLOCK);
   
   update<<<THREADS_PER_BLOCK, num_blocks>>>(randStates, cpsys_device, time);
   
   
   cudaMemcpy(psys->particles, cpsys_device->particles, sizeof(Particle) * MAX_PARTICLES, cudaMemcpyDeviceToHost);
}




