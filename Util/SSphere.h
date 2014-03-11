#pragma once
#include "SVector3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class SSphere
{

public:
  SVector3 center;
  float radius;
  SVector3 A, B, C;

  /*CUDA_CALLABLE_MEMBER SSphere();
  CUDA_CALLABLE_MEMBER SSphere(SVector3 c, float r);
  CUDA_CALLABLE_MEMBER SSphere(SVector3 A, SVector3 B, SVector3 C);
  CUDA_CALLABLE_MEMBER ~SSphere();
  CUDA_CALLABLE_MEMBER int collidesWith(SSphere other);
  int isEmpty();*/
   CUDA_CALLABLE_MEMBER SSphere()
   {
     center = SVector3();
     center.X = 0;
     center.Y = 0;
     center.Z = 0;
     radius = -999;

     this->A = SVector3();
     this->B = SVector3();
     this->C = SVector3();
   }
   CUDA_CALLABLE_MEMBER SSphere(SVector3 c, float r)
   {
     center = c;
     radius = r;

     this->A = SVector3();
     this->B = SVector3();
     this->C = SVector3();
   }

   CUDA_CALLABLE_MEMBER SSphere(SVector3 A, SVector3 B, SVector3 C)
   {
     center = SVector3();
     center.X = (A.X + B.X + C.X) / 3.0f;
     center.Y = (A.Y + B.Y + C.Y) / 3.0f;
     center.Z = (A.Z + B.Z + C.Z) / 3.0f;
     radius = ((C - A).length()) / 1.8f;

     this->A = A;
     this->B = B;
     this->C = C;
   }

   CUDA_CALLABLE_MEMBER ~SSphere() { }

   CUDA_CALLABLE_MEMBER int collidesWith(SSphere other)
   {
      if ((this->center - other.center).length() > this->radius + other.radius)
         return 0;
      return 1;
   }

   CUDA_CALLABLE_MEMBER int isEmpty()
   {
      if (A.X == 0 && B.X == 0 && C.X == 0 &&
          A.Y == 0 && B.Y == 0 && C.Y == 0 && 
          A.Z == 0 && B.Z == 0 && C.Z == 0)
      {
         return 1;
      } 
      return 0;
   }
};
