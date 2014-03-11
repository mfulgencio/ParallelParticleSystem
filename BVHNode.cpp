#include "BVHNode.h"
#include <stdlib.h>
#include <stdio.h>

BVHNode::BVHNode (SSphere end)
{
   this->hitsphere = end;
   left = NULL;
   right = NULL;
}
float max (float f1, float f2)
{
   if (f1 > f2) return f1;
   else return f2;
}

BVHNode::BVHNode (BVHNode *l, BVHNode *r)
{
   this->hitsphere = SSphere((l->hitsphere.center + r->hitsphere.center) / 2, 
                            (l->hitsphere.center - r->hitsphere.center).length() + 
                              max(r->hitsphere.radius, l->hitsphere.radius));
   left = l;
   right = r;
}
BVHNode::~BVHNode()
{

}
SSphere* BVHNode::checkHit(SSphere tocheck)
{
   if (!this->hitsphere.collidesWith(tocheck))
   {
      return NULL;
   }
   else
   {
      if (left == NULL && right == NULL)
         return &(this->hitsphere);

      SSphere* ret = left->checkHit(tocheck);
      if (ret == NULL)
      {
         ret = right->checkHit(tocheck);
      }
      return ret;
   }
}
