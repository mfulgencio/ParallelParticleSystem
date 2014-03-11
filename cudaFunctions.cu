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



__device__ int checkTriangle(SVector3 A, SVector3 B, SVector3 C, SVector3 center, float radius, SVector3 vel)
{
  SVector3 normal = ((A - C).crossProduct(A - B));
  normal /= normal.length();

  SVector3 dirA = (A - (center - (normal * -radius)));
  SVector3 dirB = (A - (center + (normal * -radius)));

  float dot1 = dirA.dotProduct(normal);
  float dot2 = dirB.dotProduct(normal);

  if (dot1 > 0 && dot2 > 0 || dot1 < 0 && dot2 < 0)
    return 0;
  return 1;
}

__global__ void collideWithBVH_kernel(Particle *particles, int num_p, BVHNode* head)
{
  /*Particle part = particles[blockIdx.x * blockDim.x + threadIdx.x];
  SSphere* hit = head->checkHit(part.sphere);
  if (hit != NULL && !hit->isEmpty() && checkTriangle(hit->A, hit->B, hit->C, particles[i].sphere.center, particles[i].sphere.radius, particles[i].velocity))
  {
    float len = particles[i].velocity.length();
    SVector3 dir = (particles[i].sphere.center) - hit->center; 
    dir /= dir.length();
    dir *= len * this->bounce;        

    particles[i].velocity = dir;
    particles[i].sphere.center += (particles[i].velocity) * this->size;
  }*/
}

extern "C" void CUDAcollideWithBVH(ParticleSystem *psys, BVHNode* head)
{
  // step 1: copy the BVH into a CUDA-compatible structure

   // step 2: copy the particles into a CUDA-compatible format
   CudaParticleSystem *cpsys_host, *cpsys_device;
   
   cudaMalloc((void**)&cpsys_host, sizeof(CudaParticleSystem));
   
   memcpy(cpsys_host->particles, psys->particles, sizeof(Particle) * MAX_PARTICLES);
   cpsys_host->numParticles = psys->numParticles;
   cpsys_host->speed = psys->speed;
   memcpy((void *)&cpsys_host->Translation, (void *)&psys->Translation, sizeof(SVector3));
   cudaMemcpy(cpsys_host, cpsys_device, sizeof(CudaParticleSystem), cudaMemcpyHostToDevice);
 
  // step 3: call the kernel

}

