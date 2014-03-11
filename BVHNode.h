#pragma once
#include "Util/SSphere.h"

class BVHNode
{
public:
   SSphere hitsphere;
   BVHNode *left, *right;

   /*BVHNode (SSphere end);
   BVHNode (BVHNode* l, BVHNode* r);
   ~BVHNode();
   SSphere* checkHit(SSphere tocheck);*/

   CUDA_CALLABLE_MEMBER BVHNode (SSphere end)
   {
      this->hitsphere = end;
      left = NULL;
      right = NULL;
   }
   CUDA_CALLABLE_MEMBER float max (float f1, float f2)
   {
      if (f1 > f2) return f1;
      else return f2;
   }

   CUDA_CALLABLE_MEMBER BVHNode (BVHNode *l, BVHNode *r)
   {
      this->hitsphere = SSphere((l->hitsphere.center + r->hitsphere.center) / 2, 
                            (l->hitsphere.center - r->hitsphere.center).length() + 
                              max(r->hitsphere.radius, l->hitsphere.radius));
      left = l;
      right = r;
   }
   CUDA_CALLABLE_MEMBER ~BVHNode()
   {

   }
    SSphere* checkHit(SSphere tocheck)
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
};
