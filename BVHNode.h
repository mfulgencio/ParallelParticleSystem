#pragma once
#include "Util/SSphere.h"

class BVHNode
{
public:
   SSphere hitsphere;
   BVHNode *left, *right;

   BVHNode (SSphere end);
   BVHNode (BVHNode* l, BVHNode* r);
   ~BVHNode();
   SSphere* checkHit(SSphere tocheck);
};
