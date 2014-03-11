#pragma once
#include "Util/SSphere.h"

#define CUDABVHSIZE 4000

typedef struct
{
   SSphere hsphere;
   int lIndex, rIndex;
} CUDA_BVH; 


class BVHNode
{
public:
   SSphere hitsphere;
   BVHNode *left, *right;

   /*BVHNode (SSphere end);
   BVHNode (BVHNode* l, BVHNode* r);
   ~BVHNode();
   SSphere* checkHit(SSphere tocheck);*/

   BVHNode (SSphere end)
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

   BVHNode (BVHNode *l, BVHNode *r)
   {
      this->hitsphere = SSphere((l->hitsphere.center + r->hitsphere.center) / 2, 
                            (l->hitsphere.center - r->hitsphere.center).length() + 
                              max(r->hitsphere.radius, l->hitsphere.radius));
      left = l;
      right = r;
   }
   ~BVHNode()
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
   // fills the bvh into an array (recursively - returns the next available index)
   int createCudaBVH(CUDA_BVH *tofill, int index)
   {
      tofill[index].hsphere = this->hitsphere;
      if (left == NULL && right == NULL)
      {
         tofill[index].lIndex = -1;
         tofill[index].rIndex = -1;
         return index + 1;
      }
      else
      {
         tofill[index].lIndex = index + 1;
         tofill[index].rIndex = this->left->createCudaBVH(tofill, index + 1);
         return this->right->createCudaBVH(tofill, tofill[index].rIndex);
      }
   }
};

