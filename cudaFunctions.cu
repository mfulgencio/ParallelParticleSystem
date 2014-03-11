/* cudaFunctions.cu
 *
 * Contains the code for all the cuda functions used
 *
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
   SVector3 Translation;
} CudaParticleSystem;


__device__ void moveParticle(Particle particle, float speed, float time) {
   particle.velocity.Y -= GRAVITY * time;

   particle.sphere.center.X += particle.velocity.X * time * speed; 
   particle.sphere.center.Y += particle.velocity.Y * time * speed; 
   particle.sphere.center.Z += particle.velocity.Z * time * speed; 
}


__device__ void resetParticle(Particle particle, float time) {

}


__global__ void update(CudaParticleSystem *cpsys, float time) {
   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
   Particle curParticle = cpsys->particles[index];
   
   moveParticle(curParticle, cpsys->speed, time);
   
   if(curParticle.sphere.center.Y < -2) {
      resetParticle(curParticle, time);
   }
}


extern "C" void cudaUpdate(ParticleSystem *psys, float time) {
   int num_blocks = 0;

   CudaParticleSystem *cpsys_host, *cpsys_device;
   
   cudaMalloc((void**)&cpsys_host, sizeof(CudaParticleSystem));
   
   memcpy(cpsys_host->particles, psys->particles, sizeof(Particle) * MAX_PARTICLES);
   cpsys_host->numParticles = psys->numParticles;
   cpsys_host->speed = psys->speed;
   memcpy((void *)&cpsys_host->Translation, (void *)&psys->Translation, sizeof(SVector3));
   
   num_blocks = ceil(MAX_PARTICLES / THREADS_PER_BLOCK);
   
   cudaMemcpy(cpsys_host, cpsys_device, sizeof(CudaParticleSystem), cudaMemcpyHostToDevice);
   
   update<<<THREADS_PER_BLOCK, num_blocks>>>(cpsys_device, time);
}




