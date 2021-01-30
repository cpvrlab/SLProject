//#############################################################################
//  File:      SLKDTree.h
//  Author:    Marcus Hudritsch
//  Date:      04-AUG-08
//  Copyright: THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifndef SLKDTREE_H
#define SLKDTREE_H

#include <stdafx.h>
#include <SLAccelStruct.h>
#include <SLKDNode.h>

//-----------------------------------------------------------------------------
//! SLKDTree implements the kd-tree space partitioning structure.
/*! A kd-tree is an axis aligned binary space partitioning (BSP) structure that
recursively splits the space on one dimension x, y or z. The art is the fast
finding of the optimal split position in a way that produce big empty cells.
*/
class SLKDTree: public SLAccelStruct 
{  public:                     
                              SLKDTree(SLMesh* m);
                             ~SLKDTree();
                             
               void           build       (SLVec3f minV, SLVec3f maxV);
               void           updateStats (SLGroup* parent);
               void           draw        (SLSceneView* sv);
               SLbool         intersect   (SLRay* ray);
               
               void           buildTree   (SLint maxDepth,
                                           SLVec3f minV, SLVec3f maxV);
               void           splitNode   (SLKDNode* node, 
                                           SLint depth, SLint maxDepth, 
                                           SLVec3f minV, SLVec3f maxV);
               void           findSplit1  (SLKDNode* node, SLint depth,
                                           SLVec3f minV, SLVec3f maxV);
               void           findSplit2  (SLKDNode* node, SLint depth,
                                           SLVec3f minV, SLVec3f maxV);
               void           findSplit3  (SLKDNode* node, SLint depth,
                                           SLVec3f minV, SLVec3f maxV);
               void           drawNode    (SLKDNode* node, SLint depth,
                                           SLVec3f minV, SLVec3f maxV);
   private:
               SLKDNode*	   _kdRoot;       //!< pointer to root kd-node
               SLbool         _kdDump;       //!< flag for tree dump
               SLuint         _kdNodeCnt;    //!< Num. of leaf nodes
               SLint          _kdMaxDepth;   //!< max. depth of kd splits
               SLKdTriaAABB*  _kdTriaAB;     //!< array with aabb's of triangles
};
//-----------------------------------------------------------------------------
#endif //SLKDTREE_H

