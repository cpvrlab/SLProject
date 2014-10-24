//#############################################################################
//  File:      SLKDNode.h
//  Author:    Original version from the students Roll, Schaub & Stocker
//  Date:      02-APR-06
//  Copyright: Marcus Hudritsch, Kirchrain 18, 2572 Sutz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifndef SLKDNODE_H
#define SLKDNODE_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
// Maximum of the stack used in the Kd-traversal code
#define MAXSTACKDEPTH 50
#define XAXIS 0
#define YAXIS 1
#define ZAXIS 2
#define NOAXIS 3

//-----------------------------------------------------------------------------
struct SLKdTriaAABB
{  SLVec3f min;   //!< minimal corner of AABB
   SLVec3f max;   //!< maximal corner of AABB
   SLfloat sa;    //!< surface area
};
//-----------------------------------------------------------------------------
class SLKDNode
{  public:
                     SLKDNode()
                     {  child1 = 0;
                        child2 = 0;
                        splitAxis = NOAXIS;
                        kdNodeCnt++;
                     }
                    ~SLKDNode()
                     {  if (child1) delete child1;
                        if (child2) delete child2;
                     }
                     
      void           draw(SLVec3f minV, SLVec3f maxV);
      
      SLKDNode*      child1;
      SLKDNode*      child2;
      SLVushort      tria;
      SLuchar        splitAxis; //0=x, 1=y, 2=z and 3=noAxis=leaf node
      SLfloat        splitDist;
      static SLuint  kdNodeCnt;
};
//-----------------------------------------------------------------------------
// Entry for stack operation.
struct SLKdStackElem
{  SLKDNode* node;   // pointer to far child
   SLfloat   t;      // the entry/exit signed distance
   SLVec3f   p;      // the coordinates of entry/exit point
   SLint     prev;   // the pointer to the previous stack item
};
//-----------------------------------------------------------------------------
#endif // SLKDNODE_H