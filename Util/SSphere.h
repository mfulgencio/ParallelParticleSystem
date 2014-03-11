#pragma once
#include "SVector3.h"

class SSphere
{

public:
  SVector3 center;
  float radius;
  SVector3 A, B, C;

  SSphere();
  SSphere(SVector3 c, float r);
  SSphere(SVector3 A, SVector3 B, SVector3 C);
  ~SSphere();
  int collidesWith(SSphere other);
  int isEmpty();
};
