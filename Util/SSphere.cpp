

#include "SSphere.h"

SSphere::SSphere()
{
  center = SVector3();
  center.X = 0;
  center.Y = 0;
  center.Z = 0;
  radius = -999;
}
SSphere::SSphere(SVector3 c, float r)
{
  center = c;
  radius = r;
}

SSphere::SSphere(SVector3 A, SVector3 B, SVector3 C)
{
  center = SVector3();
  center.X = (A.X + B.X + C.X) / 3.0f;
  center.Y = (A.Y + B.Y + C.Y) / 3.0f;
  center.Z = (A.Z + B.Z + C.Z) / 3.0f;
  radius = ((C - A).length()) / 1.8f;
}

SSphere::~SSphere() { }

int SSphere::collidesWith(SSphere other)
{
   if ((this->center - other.center).length() > this->radius + other.radius)
      return 0;
   return 1;
}

