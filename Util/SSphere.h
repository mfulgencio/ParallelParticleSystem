#pragma once
#include "SVector3.h"

class SSphere
{

public:
  SVector3 center;
  float radius;

  SSphere();
  SSphere(SVector3 A, SVector3 B, SVector3 C);
  ~SSphere();
  int collidesWith(SSphere* other);

};
