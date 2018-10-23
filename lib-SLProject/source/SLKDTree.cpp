//#############################################################################
//  File:      SLKDTree.cpp
//  Author:    Marcus Hudritsch
//  Date:      04-AUG-08
//  Copyright: THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include <SLGroup.h>
#include <SLMesh.h>
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLKDNode.h>
#include <SLKDTree.h>

//-----------------------------------------------------------------------------
SLuint SLKDNode::kdNodeCnt = 0;
//-----------------------------------------------------------------------------
#define MIN_FACES_FOR_ACCELERATION 3
//-----------------------------------------------------------------------------
SLKDTree::SLKDTree(SLMesh* m) : SLAccelStruct(m)
{   
   _voxCnt      = 0;
   _voxCntEmpty = 0;
   _voxMaxTria  = 0;
   _voxAvgTria  = 0;
   _kdRoot      = 0;
   _kdMaxDepth  = 0;  
}
//-----------------------------------------------------------------------------
SLKDTree::~SLKDTree()
{  
   if (_kdRoot) delete _kdRoot;
}
//-----------------------------------------------------------------------------
/*! SLKDTree::draw implements the abstact draw method from SLAccelStruct
*/
void SLKDTree::draw(SLSceneView* sv)
{  
   //if (sv->drawingBits() & SL_DB_VOXELS || _m->drawingBits() & SL_DB_VOXELS)
   //{  
   //   // push OGL matrix & attributes
   //   glPushMatrix();
   //   glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT);
   //   glDisable(GL_LIGHTING);
   //   glDisable(GL_TEXTURE_2D);
   //   
   //   // set the initial camera view
   //   sv->camera()->setView();
   //   
   //   // call the recursive kd-tree drawing method 
   //   drawNode(_kdRoot, 0, _minV, _maxV);  
   //    
   //   // pop back OGL matrix & attributes
   //   glEnable(GL_LIGHTING);
   //   glPopAttrib();
   //   glPopMatrix();
   //}
}
//-----------------------------------------------------------------------------
/*! Updates the parent groups statistics.
*/
void SLKDTree::updateStats(SLGroup* parent)
{  
   parent->numVoxels    += _voxCnt;
   parent->numVoxEmpty  += _voxCntEmpty;
   if (_voxMaxTria > parent->numVoxMaxTria)
      parent->numVoxMaxTria = _voxMaxTria;
}
//-----------------------------------------------------------------------------
/*! SLKDTree::build starts the kd-tree building by calling the recursive 
SLKDTree::buildTree method.
*/
void SLKDTree::build(SLVec3f minV, SLVec3f maxV)
{  
   _minV = minV;
   _maxV = maxV;
   buildTree(1, minV, maxV);
}

//-----------------------------------------------------------------------------
/*! 
SLKDTree::intersect implements the kd-tree traversale algorithm TA_B_rec
from Vlasimil Havran's thesis "Heuristic Ray Schooting Algorithms". The 
recursion is removed with a stack with elements of SLKdStackElem.
*/
SLbool SLKDTree::intersect(SLRay* ray)
{  if (_kdRoot)
   {  
      SLfloat tmin = 0, tmax = 0; // hit distance to aabb
      if (_m->aabb().isHit(ray, tmin, tmax))
      {  
         SLfloat t; // signed distance to the splitting plane
         SLVec3f O = ray->origin;
         SLVec3f D = ray->dir;
         
         // stack required for traversal to store far children
         SLKdStackElem stack[MAXSTACKDEPTH];
         
         SLKDNode* farChild;
         SLKDNode* curNode = _kdRoot;  // start from the kd-tree root node
         SLint     enPt    = 0;        // setup initial entry point
         SLint     exPt    = 1;        // pointer to the stack

         stack[enPt].t = tmin;   // set the signed distance

         // distinguish between internal and external origin
         if (tmin >= 0.0) 
              stack[enPt].p.add(O,D*tmin);   // a ray with external origin
         else stack[enPt].p = O;             // a ray with internal origin
                
         // setup initial exit point in the stack
         stack[exPt].t = tmax;
         stack[exPt].p.add(O,D*tmax);
         stack[exPt].node = 0; // set termination flag

         // loop through the whole kd-tree, until an object is intersected 
         // or ray leaves the scene
         while (curNode)
         {
            // loop until a nonempty leaf is found 
            while (curNode->splitAxis != NOAXIS)
            {
               // retrieve axis and position of splitting plane
               SLfloat splitDist = curNode->splitDist;
               SLint   axis      = curNode->splitAxis;     
               
               // check next node to descend
               if (stack[enPt].p[axis] <= splitDist)
               {  if (stack[exPt].p[axis] <= splitDist)
                  {  // case N1, N2, N3, P5, Z2, and Z3
                     curNode = curNode->child1;   
                     continue;
                  }
                  if (stack[exPt].p[axis] == splitDist)
                  {  // case Z1
                     curNode = curNode->child2;  
                     continue; 
                  }

                  // case N4
                  farChild = curNode->child2;
                  curNode  = curNode->child1;
               }
               else
               {  // (stack[enPt].p[axis] > splitDist)
                  if (splitDist < stack[exPt].p[axis])
                  {  // case P1, P2, P3, and N5
                     curNode = curNode->child2;  
                     continue;
                  }

                  // case P4
                  farChild = curNode->child1;
                  curNode  = curNode->child2;
               }

               // case P4 or N4 -> traverse both children: add nodes to stack
               // signed distance to the splitting plane
               t = (splitDist - O[axis]) / D[axis];

               // setup the new exit point
               SLint tmp = exPt;
               exPt++;

               // possibly skip current entry point so not to overwrite the data
               if (exPt == enPt) exPt++;

               // push values onto the stack
               SLuint axis1 = NEXTAXIS(axis);
               SLuint axis2 = NEXTAXIS(axis1);
               stack[exPt].prev = tmp;
               stack[exPt].t = t;
               stack[exPt].node = farChild;
               stack[exPt].p[axis]  = splitDist;         
               stack[exPt].p[axis1] = O[axis1] + t*D[axis1];
               stack[exPt].p[axis2] = O[axis2] + t*D[axis2];
            }

            // current node is the leaf, empty or full
            // intersect ray with each object in the object list, discarding
            // those lying before stack[enPt].t or farther than stack[exPt].t
            if (curNode->tria.size() > 0)
            {  for(SLuint i=0; i<curNode->tria.size(); i++)
                  _m->hitTriangleOS(ray, curNode->tria[i]);
      	         
               if (ray->length > stack[enPt].t && 
                   ray->length <= stack[exPt].t) return;
            }
            
            // get next node from stack
            enPt = exPt; // the signed distance intervals are adjacent
            curNode = stack[exPt].node;
            exPt = stack[enPt].prev;
         }
      }
   } 
   else // not enough trias to build tree
   {  SLfloat tmin = 0, tmax = 0; // hit distance to aabb
      if (_m->aabb().isHit(ray, tmin, tmax))
      {  for (SLuint t=0; t<_m->numF; ++t)
         {  _m->hitTriangleOS(ray, t);
         }
      }
   }
}
//-----------------------------------------------------------------------------
/*! 
SLKDTree::buildTree builds a kd tree by recursively splitting up its AABB
*/
void SLKDTree::buildTree(SLint maxDepth, SLVec3f minV, SLVec3f maxV)
{
   _kdDump = false;
   maxDepth = 100;
   
   // create new tree
   SLKDNode::kdNodeCnt = 0;
   _kdMaxDepth = 0;
   if (_kdRoot) delete _kdRoot;
   _kdRoot = new SLKDNode();
   _kdRoot->tria.reserve(_m->numF);
   
   _kdTriaAB = new SLKdTriaAABB[_m->numF];

   //add all polygons to the root node
   for(SLuint i=0; i<_m->numF; i++)
   {  _kdRoot->tria.push_back(i);
      // for findSplit3 calculate AABB and surface area  
      _m->P[_m->F[i].iA].x;    
      _kdTriaAB[i].min.x = SL_min(_m->P[_m->F[i].iA].x, 
                                  _m->P[_m->F[i].iB].x, 
                                  _m->P[_m->F[i].iC].x);
      _kdTriaAB[i].min.y = SL_min(_m->P[_m->F[i].iA].y, 
                                  _m->P[_m->F[i].iB].y, 
                                  _m->P[_m->F[i].iC].y);
      _kdTriaAB[i].min.z = SL_min(_m->P[_m->F[i].iA].z, 
                                  _m->P[_m->F[i].iB].z, 
                                  _m->P[_m->F[i].iC].z);
      _kdTriaAB[i].max.x = SL_max(_m->P[_m->F[i].iA].x, 
                                  _m->P[_m->F[i].iB].x, 
                                  _m->P[_m->F[i].iC].x);
      _kdTriaAB[i].max.y = SL_max(_m->P[_m->F[i].iA].y, 
                                  _m->P[_m->F[i].iB].y, 
                                  _m->P[_m->F[i].iC].y);
      _kdTriaAB[i].max.z = SL_max(_m->P[_m->F[i].iA].z, 
                                  _m->P[_m->F[i].iB].z, 
                                  _m->P[_m->F[i].iC].z);
      
      SLVec3f size(_kdTriaAB[i].max - _kdTriaAB[i].min);
      _kdTriaAB[i].sa = 2*(size.x*size.y + size.x*size.z + size.y*size.z);
   }
   
   if (_kdDump) printf("\n\nNew kdTree (max. depth: %d)\n", maxDepth);
   
   splitNode(_kdRoot, 0, maxDepth, minV, maxV);
   
   if (_kdDump) printf("\n\nMax. depth reached: %d\n", _kdMaxDepth);
   
   // delete aabb's of triangles
   if (_kdTriaAB) delete _kdTriaAB;
}
//-----------------------------------------------------------------------------
/*! 
SLKDTree::splitNode splits a kd node into 2 children and distributes the 
triangles to them. The core method is one of the spit position finding routine
kdFindSplit methods
*/
void SLKDTree::splitNode(SLKDNode* node, SLint depth, SLint maxDepth,
                         SLVec3f minV, SLVec3f maxV)
{
   static SLint i = 0;
   static SLuchar axis = 0;
   static const SLuint MINTRIAS = 10;
 
   if (depth > _kdMaxDepth) _kdMaxDepth = depth;
   
   if (depth >= maxDepth)
   {  if (node->tria.size()>_voxMaxTria) _voxMaxTria = node->tria.size();
      if (node->tria.size()==0) _voxCntEmpty++;
      _voxCnt++;
      return;
   }
   
   i++;
   
   if (_kdDump) printf("Split %d: ", i);
   
   if(node->tria.size() < MINTRIAS) 
   {  if (_kdDump) printf("Leaf: %d\n", node->tria.size());
      if (node->tria.size()>_voxMaxTria) _voxMaxTria = node->tria.size();
      if (node->tria.size()==0) _voxCntEmpty++;
      _voxCnt++;
      return;
   }
   
   // keep the parent size before
   SLuint parentSize = node->tria.size();
   
   //////////////////////////////////////
   findSplit1(node, depth, minV, maxV);
   //////////////////////////////////////
   
   // Set min & max corners of child nodes
   SLVec3f minV1 = minV, minV2 = minV;
   SLVec3f maxV1 = maxV, maxV2 = maxV;
   if(node->splitAxis == XAXIS)  maxV1.x = minV2.x = node->splitDist; else 
   if(node->splitAxis == YAXIS)  maxV1.y = minV2.y = node->splitDist; else 
   if(node->splitAxis == ZAXIS)  maxV1.z = minV2.z = node->splitDist;
   else printf("Error\n");
   
   // Calculate voxel extention and center for triangle box overlap test
   SLVec3f voxExt1, voxExt2, center1, center2;
   voxExt1.sub(maxV1, minV1); voxExt1*=0.5f;
   voxExt2.sub(maxV2, minV2); voxExt2*=0.5f;
   center1.add(minV1, voxExt1);
   center2.add(minV2, voxExt2);
   SLfloat vert[3][3];
      
   // Create new children
   node->child1 = new SLKDNode();
   node->child2 = new SLKDNode();
     
   // loop through all triangle and add them to the left or right or both children
   for(SLuint i=0; i<node->tria.size(); ++i)
   {  SLushort iT = node->tria[i];
      // Copy triangle vertices into SLfloat array[3][3]
      vert[0][0] = _m->P[_m->F[iT].iA].x;
      vert[0][1] = _m->P[_m->F[iT].iA].y; 
      vert[0][2] = _m->P[_m->F[iT].iA].z;
      vert[1][0] = _m->P[_m->F[iT].iB].x; 
      vert[1][1] = _m->P[_m->F[iT].iB].y; 
      vert[1][2] = _m->P[_m->F[iT].iB].z;
      vert[2][0] = _m->P[_m->F[iT].iC].x; 
      vert[2][1] = _m->P[_m->F[iT].iC].y; 
      vert[2][2] = _m->P[_m->F[iT].iC].z;
   
      if (triBoxOverlap(center1, voxExt1, vert))
      {  node->child1->tria.push_back(node->tria[i]);
      } 
      if(triBoxOverlap(center2, voxExt2, vert))
      {  node->child2->tria.push_back(node->tria[i]);
      }
   }
     
   // all trias are now in the childs so empty now the parent
   node->tria.clear();
   
   if (_kdDump)
   {  for(SLint t=0; t<depth; ++t) printf(" ");
      printf("parent: %d, left: %d, right: %d\n", 
              node->tria.size(),
              node->child1->tria.size(),
              node->child2->tria.size());
   }
   
   // termination if parent has the same size as child1
   if (node->child1->tria.size() == parentSize &&
       node->child2->tria.size() == parentSize)
   {  if (_kdDump) printf("can't split anymore\n");
      if (node->child1->tria.size()>_voxMaxTria) _voxMaxTria = node->child1->tria.size();
      if (node->child2->tria.size()>_voxMaxTria) _voxMaxTria = node->child2->tria.size();
      _voxCnt+=2;
      return;
   }
     
   // split children
   splitNode(node->child1, depth+1, maxDepth, minV1, maxV1);
   splitNode(node->child2, depth+1, maxDepth, minV2, maxV2);
}
//-----------------------------------------------------------------------------
/*! 
SLKDTree::findSplit1 splits at center of the node and on its biggest axis
*/
void SLKDTree::findSplit1(SLKDNode* node, SLint depth, 
                          SLVec3f minV, SLVec3f maxV)
{  
   // Splitpoint is in the middle
   SLVec3f midPt;
   midPt.add(minV, maxV);
   midPt *= 0.5f;
   
   // Split axis is the one that has the biggest extent
   SLVec3f size(maxV-minV);
   node->splitAxis = size.maxAxis();
   node->splitDist = midPt[node->splitAxis];
}
//-----------------------------------------------------------------------------
/*! 
SLKDTree::findSplit2 splits in the center if there is no empty space. First 
the empty axis aligned space is calculated with AABB around the triangles. 
*/
void SLKDTree::findSplit2(SLKDNode* node, SLint depth, 
                          SLVec3f minV, SLVec3f maxV)
{  
   SLVec3f newMin, newMax, newSize, newOffset;
   
   if (node != _kdRoot && depth<5)
   {  // Init new min & max corners
      newMin.set( SL_REAL_MAX,  SL_REAL_MAX,  SL_REAL_MAX);
      newMax.set(-SL_REAL_MAX, -SL_REAL_MAX, -SL_REAL_MAX);

      // calc new min & max off all triangles inside the node
      for(SLuint i=0; i<node->tria.size(); ++i)
      {  SLVec3f VA(_m->P[_m->F[node->tria[i]].iA]);
         SLVec3f VB(_m->P[_m->F[node->tria[i]].iB]);
         SLVec3f VC(_m->P[_m->F[node->tria[i]].iC]);
         if (VA.x < newMin.x) newMin.x = VA.x;
         if (VA.x > newMax.x) newMax.x = VA.x;
         if (VA.y < newMin.y) newMin.y = VA.y;
         if (VA.y > newMax.y) newMax.y = VA.y;
         if (VA.z < newMin.z) newMin.z = VA.z;
         if (VA.z > newMax.z) newMax.z = VA.z;
         if (VB.x < newMin.x) newMin.x = VB.x;
         if (VB.x > newMax.x) newMax.x = VB.x;
         if (VB.y < newMin.y) newMin.y = VB.y;
         if (VB.y > newMax.y) newMax.y = VB.y;
         if (VB.z < newMin.z) newMin.z = VB.z;
         if (VB.z > newMax.z) newMax.z = VB.z;
         if (VC.x < newMin.x) newMin.x = VC.x;
         if (VC.x > newMax.x) newMax.x = VC.x;
         if (VC.y < newMin.y) newMin.y = VC.y;
         if (VC.y > newMax.y) newMax.y = VC.y;
         if (VC.z < newMin.z) newMin.z = VC.z;
         if (VC.z > newMax.z) newMax.z = VC.z;
      }
   
      newSize.sub(newMax, newMin);
      newOffset = newSize*0.3f;
      newMin -= newOffset;
      newMax += newOffset;
   } 
   else
   {  // for the parent node set the newMin-Max to its AABB
      newMin = minV;
      newMax = maxV;
   }
   
   // Split in the middle if newMinMax is equal to the parent
   // This means that there is no empty axis aligned space
   if (newMin <= minV && newMax >= maxV)
   {  
      // Splitpoint in the middle
      SLVec3f midPt;
      midPt.add(minV, maxV);
      midPt *= 0.5f;
      
      // Split axis is the one that has the biggest extent
      SLVec3f size(maxV-minV);
      node->splitAxis = size.maxAxis();
      node->splitDist = midPt[node->splitAxis];
   } 
   else // if new MinMax is smaller that the parents MinMax there is empty space
   {
      SLVec3f diffMin, diffMax;
      diffMin.sub(newMin, minV);
      diffMax.sub(maxV, newMax);
      SLint   maxAxisMin, maxAxisMax;
      SLfloat maxDiffMin = diffMin.maxXYZ(maxAxisMin);
      SLfloat maxDiffMax = diffMax.maxXYZ(maxAxisMax);
      
      if (maxDiffMin > maxDiffMax)
      {  node->splitAxis = maxAxisMin;
         node->splitDist = newMin[maxAxisMin];
      
      } else
      {  node->splitAxis = maxAxisMax;
         node->splitDist = newMax[maxAxisMax];
      }      
   }	
}

//-----------------------------------------------------------------------------
/*! 
SLKDTree::findSplit3: not finished
*/
void SLKDTree::findSplit3(SLKDNode* node, SLint depth, 
                          SLVec3f minV, SLVec3f maxV)
{  
   // calculate size of parent node and the deltas of the bins
   SLuint numBin = 16;
   SLVec3f size(maxV-minV);
   SLVec3f delta(size/(SLfloat)numBin);
   
   // Loof over all axis & calculate minimal cost
   for (SLuint axis=XAXIS; axis<=ZAXIS; ++axis)
   {
      // allocate bins for counting AABB min & max 
      SLVuint* binMin = new SLVuint[numBin];
      SLVuint* binMax = new SLVuint[numBin];
   
      ////////////////////////////////////////
      // STEP 1: Fill in the min & max bins //
      ////////////////////////////////////////
      
      // loop over triangles of bin & increment start and end bin
      for(SLuint i=0; i<node->tria.size(); ++i)
      {  
         // loop through bins & search the triangles min & max bin
         SLfloat minBound = minV[axis];
         SLbool  foundMin = false;
         for (SLuint b=0; b<numBin; ++b)
         {  // check if the triangle minimum is in this bin
            if (!foundMin && 
                _kdTriaAB[i].min[axis] > minBound && 
                _kdTriaAB[i].min[axis] <= minBound+delta[axis])
            {  binMin[b].push_back(i);    // add the index of the triangle to this bin
               foundMin = true;           // don't look for the minimum anymore     
            }
            // check if the triangle maximum is in this bin
            if ( foundMin && 
                _kdTriaAB[i].max[axis] > minBound && 
                _kdTriaAB[i].max[axis] <= minBound+delta[axis])
            {  binMax[b].push_back(i);   // add the index of the triangle to this bin
               break;                     // found min & max
            }
            minBound += delta[axis];      // increment the left boundry of the bin
         }
      }
      
      ///////////////////////////////////////////////////////
      // STEP 2: Calculate the SAH on all in bin boundries //
      ///////////////////////////////////////////////////////
      
      SLfloat* cost    = new SLfloat[numBin-1];   // SAH cost at x-boundries
      SLfloat* SALeft  = new SLfloat[numBin-1];   // surface are to the left at x-boundries
      SLfloat* SARight = new SLfloat[numBin-1];   // surface are to the right at x-boundries
      
      for (SLuint b=0; b<numBin-1; ++b)
      {  
         cost[b] = 0;
         SALeft[b] = 0;
         SARight[b] = 0;
         
         // sum up SA of all AABB that have their max to the left
         SLuint leftBin;
         for (leftBin=0; leftBin<b+1; ++leftBin)
         {  for (SLuint i=0; i<binMax[leftBin].size()-1; ++i)
            {  SALeft[b] += _kdTriaAB[binMax[leftBin][i]].sa;
            }
         }
         // add the SA of all AABB the have their min in the bin to the left
         for (SLuint i=0; i<binMax[leftBin].size()-1; ++i)
         {  SALeft[b] += _kdTriaAB[binMax[leftBin][i]].sa;
         }
         
      }
      
      //////////////
      // clean up //
      //////////////
      
      if (cost)    delete cost;
      if (SALeft)  delete SALeft;
      if (SARight) delete SARight;
      if (binMin)  delete binMin;
      if (binMax)  delete binMax;
      
   } // for loop axis
}

//-----------------------------------------------------------------------------
/*! 
SLKDTree::drawNode recursively draws all split planes of all nodes
*/
void SLKDTree::drawNode(SLKDNode* node, SLint depth, 
                        SLVec3f minV, SLVec3f maxV)
{  if (node==0) return;
   if (node->splitAxis==NOAXIS) return;
   
   // draw split plane
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBegin(GL_QUADS);
   if (node->splitAxis==XAXIS) 
   {  glColor4f(1, 0, 1, 0.5f); 
      // right rect
      glVertex3r(node->splitDist, minV.y, minV.z);
      glVertex3r(node->splitDist, maxV.y, minV.z);
      glVertex3r(node->splitDist, maxV.y, maxV.z);
      glVertex3r(node->splitDist, minV.y, maxV.z);
   } else if (node->splitAxis==YAXIS) 
   {  glColor4f(1, 1, 0, 0.5f); 
      glVertex3r(minV.x, node->splitDist, minV.z);
      glVertex3r(maxV.x, node->splitDist, minV.z);
      glVertex3r(maxV.x, node->splitDist, maxV.z);
      glVertex3r(minV.x, node->splitDist, maxV.z);
   } else if (node->splitAxis==ZAXIS)
   {  glColor4f(0, 1, 1, 0.5f);
      glVertex3r(minV.x, minV.y, node->splitDist); 
      glVertex3r(minV.x, maxV.y, node->splitDist);
      glVertex3r(maxV.x, maxV.y, node->splitDist);
      glVertex3r(maxV.x, minV.y, node->splitDist);
   } 
   glEnd();
   glDisable(GL_BLEND);
   glEnable(GL_CULL_FACE);
   
   // Set min & max corners of child nodes
   SLVec3f minV1 = minV;
   SLVec3f minV2 = minV;
   SLVec3f maxV1 = maxV;
   SLVec3f maxV2 = maxV;
   if(node->splitAxis == 0) 
      maxV1.x = minV2.x = node->splitDist; 
   else if(node->splitAxis == 1)  
      maxV1.y = minV2.y = node->splitDist; 
   else if(node->splitAxis == 2)
      maxV1.z = minV2.z = node->splitDist;
   
   // draw children by recurse
   drawNode(node->child1, depth+1, minV1, maxV1);
   drawNode(node->child2, depth+1, minV2, maxV2);   
}
//-----------------------------------------------------------------------------
