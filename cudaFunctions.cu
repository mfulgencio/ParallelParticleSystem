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
#include "BVHNode.h"

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


__device__ void moveParticle(Particle *particle, float speed, float time) {
   particle->velocity.Y -= GRAVITY * time;

   particle->sphere.center.X += particle->velocity.X * time * speed; 
   particle->sphere.center.Y += particle->velocity.Y * time * speed; 
   particle->sphere.center.Z += particle->velocity.Z * time * speed; 
}


__device__ void resetParticle(curandState *randStates, Particle *particle, SVector3 Translation, float random, float time) {
   particle->sphere.center.X = Translation.X + rbFloat(randStates) * random;
   particle->sphere.center.Y = Translation.Y + rbFloat(randStates) * random;
   particle->sphere.center.Z = Translation.Z + rbFloat(randStates) * random;
   
   particle->velocity.X = rbFloat(randStates) * SPEED;
   particle->velocity.Y = -rFloat(randStates) * YSPEED;
   particle->velocity.Z = rbFloat(randStates) * SPEED;
}


__global__ void update(curandState *randStates, CudaParticleSystem *cpsys, float time) {
   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
   Particle curParticle = cpsys->particles[index];
   
   moveParticle(&curParticle, cpsys->speed, time);
   
   if(curParticle.sphere.center.Y < -2) {
      curand_init(1234, index, 0, &randStates[index]);
      resetParticle(randStates, &curParticle, cpsys->Translation, cpsys->random, time);
   }
   
   cpsys->particles[index] = curParticle;
}


extern "C" void cudaUpdate(ParticleSystem *psys, float time) {
   int num_blocks = 0;
   curandState *randStates;

   CudaParticleSystem *cpsys_device;
   
   cudaMalloc((void **)&cpsys_device, sizeof(CudaParticleSystem));
   cudaMalloc((void **)&randStates, psys->numParticles * sizeof(curandState));
   
   cudaMemcpy(cpsys_device->particles, psys->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->numParticles, &psys->numParticles, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->speed, &psys->speed, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->random, &psys->random, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->Translation, &psys->Translation, sizeof(SVector3), cudaMemcpyHostToDevice);
   
   num_blocks = psys->numParticles / THREADS_PER_BLOCK + 1;
   
   update<<<THREADS_PER_BLOCK, num_blocks>>>(randStates, cpsys_device, time);

   cudaMemcpy(psys->particles, cpsys_device->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyDeviceToHost);
   
   cudaFree(cpsys_device);
   cudaFree(randStates);
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


__device__ SSphere* checkHit(CUDA_BVH* bvh, SSphere sphere)
{
   int queue[CUDABVHSIZE];
   queue[0] = 0; // add the head of the bvh to the queue
   int start = 0, end = 1;

   while (start != end) // while the queue isn't empty
   {
      SSphere tocheck = bvh[queue[start]].hsphere;
      if (tocheck.collidesWith(sphere))
      { 
         if (bvh[queue[start]].lIndex == -1 && bvh[queue[start]].rIndex == -1) // we found a leaf
            return &bvh[queue[start]].hsphere;
         // else
         queue[end++] = bvh[queue[start]].lIndex;
         queue[end++] = bvh[queue[start]].rIndex;
      }

      start++;
   }
   return NULL;
}

__global__ void collideWithBVH_kernel(CudaParticleSystem *cpsys, int num_p, CUDA_BVH* bvh, float bounce, float size)
{
  Particle part = cpsys->particles[blockIdx.x * blockDim.x + threadIdx.x];
  SSphere* hit = checkHit(bvh, part.sphere);

  if (hit != NULL && !hit->isEmpty() && checkTriangle(hit->A, hit->B, hit->C, part.sphere.center, part.sphere.radius, part.velocity))
  {
    float len = part.velocity.length();
    SVector3 dir = (part.sphere.center) - hit->center; 
    dir /= dir.length();
    dir *= len * bounce;        

    part.velocity = dir;
    part.sphere.center += (part.velocity) * size;
  }
  
  cpsys->particles[blockIdx.x * blockDim.x + threadIdx.x] = part;
}

extern "C" void CUDAcollideWithBVH(ParticleSystem *psys, CUDA_BVH* bvh)
{
   // step 1: copy the particles into a CUDA-compatible format
   CudaParticleSystem *cpsys_device;
   CUDA_BVH *cuda_bvh;
   
   cudaMalloc((void **)&cpsys_device, sizeof(CudaParticleSystem));
   cudaMalloc((void **)&cuda_bvh, CUDABVHSIZE * sizeof(CUDA_BVH));
   
   cudaMemcpy(cuda_bvh, bvh, CUDABVHSIZE * sizeof(CUDA_BVH), cudaMemcpyHostToDevice);
   cudaMemcpy(cpsys_device->particles, psys->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->numParticles, &psys->numParticles, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->speed, &psys->speed, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->random, &psys->random, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->Translation, &psys->Translation, sizeof(SVector3), cudaMemcpyHostToDevice);
 
   // step 2: call the kernel
   int num_blocks = psys->numParticles / THREADS_PER_BLOCK + 1;
   collideWithBVH_kernel<<<THREADS_PER_BLOCK, num_blocks>>>(cpsys_device, psys->numParticles, cuda_bvh, psys->bounce, psys->size);
   cudaMemcpy(psys->particles, cpsys_device->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyDeviceToHost);
   
   cudaFree(cpsys_device);
   cudaFree(cuda_bvh);
}

__global__ void collideWith_kernel(CudaParticleSystem *cpsys, SSphere* spheres, int numspheres, float bounce, float size) {
  Particle part = cpsys->particles[blockIdx.x * blockDim.x + threadIdx.x];

  for (int j = 0; j < numspheres; j++)
  {
    if (spheres[j].collidesWith(part.sphere) && 
        checkTriangle(spheres[j].A, spheres[j].B, spheres[j].C, part.sphere.center, part.sphere.radius, part.velocity))
    {
      float len = part.velocity.length();
      SVector3 dir = (part.sphere.center) - spheres[j].center; 
      dir /= dir.length();
      dir *= len * bounce;        

      part.velocity = dir;
      part.sphere.center += (part.velocity) * size;

      break;
    }
  }

  cpsys->particles[blockIdx.x * blockDim.x + threadIdx.x] = part;
}

extern "C" void CUDAcollideWith(ParticleSystem *psys, std::vector<SSphere> spheres) {
   CudaParticleSystem *cpsys_device;
   SSphere *cu_spheres;
   
   cudaMalloc((void **)&cpsys_device, sizeof(CudaParticleSystem));
   cudaMalloc((void **)&cu_spheres, sizeof(SSphere) * spheres.size());
   
   cudaMemcpy(cpsys_device->particles, psys->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->numParticles, &psys->numParticles, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->speed, &psys->speed, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->random, &psys->random, sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(&cpsys_device->Translation, &psys->Translation, sizeof(SVector3), cudaMemcpyHostToDevice);
   cudaMemcpy(cu_spheres, &spheres[0], sizeof(SSphere) * spheres.size(), cudaMemcpyHostToDevice);
   
 
   int num_blocks = psys->numParticles / THREADS_PER_BLOCK + 1;
   collideWith_kernel<<<THREADS_PER_BLOCK, num_blocks>>>(cpsys_device, cu_spheres, spheres.size(), psys->bounce, psys->size);
   cudaMemcpy(psys->particles, cpsys_device->particles, sizeof(Particle) * psys->numParticles, cudaMemcpyDeviceToHost);

   cudaFree(cpsys_device);
   cudaFree(cu_spheres);
}
