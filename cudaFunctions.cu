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

#define THREADS_PER_BLOCK 32

// Variables needed for moveParticle(): ParticleSystem.particles[], ParticleSystem.speed,
   // particle index, float "time" variable, GRAVITY constant
// Variables needed for resetParticle(): ParticleSystem.particles[], particle index,
   // SVector3 translation, rbFloat() & rFloat(), SPEED & YSPEED constants
   
typedef struct {
   Particle particles[MAX_PARTICLES];
   int numParticles;
   SVector3 Translation;
} CudaParticleSystem;


__global__ void update(CudaParticleSystem *cpsys, float time) {
   
}


extern "C" void cudaUpdate(ParticleSystem *psys, float time) {
   int num_blocks = 0;

   CudaParticleSystem *cpsys_host, *cpsys_device;
   
   cudaMalloc((void**)&cpsys_host, sizeof(CudaParticleSystem));
   
   memcpy(cpsys_host->particles, psys->particles, sizeof(Particle) * MAX_PARTICLES);
   cpsys_host->numParticles = psys->numParticles;
   memcpy((void *)&cpsys_host->Translation, (void *)&psys->Translation, sizeof(SVector3));
   
   num_blocks = ceil(MAX_PARTICLES / THREADS_PER_BLOCK);
   
   cudaMemcpy(cpsys_host, cpsys_device, sizeof(CudaParticleSystem), cudaMemcpyHostToDevice);
   
   update<<<THREADS_PER_BLOCK, num_blocks>>>(cpsys_device, time);
}




