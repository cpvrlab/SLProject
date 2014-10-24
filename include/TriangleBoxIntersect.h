//-----------------------------------------------------------------------------
// AABB-triangle overlap test code                      
// by Tomas Möller                                      
// Function: int triBoxOverlap(float boxcenter[3],      
//          float boxhalfsize[3],float triverts[3][3]); 
// History:                                             
//   2001-03-05: released the code in its first version 
//                                                      
// Acknowledgement: Many thanks to Pierre Terdiman for  
// suggestions and discussions on how to optimize code. 
// Thanks to David Hunt for finding a ">="-bug!         
//-----------------------------------------------------------------------------

#ifndef TRIANGLEBOXOVERLAP_H
#define TRIANGLEBOXOVERLAP_H

#include <math.h>
#include <stdio.h>

#define kX 0
#define kY 1
#define kZ 2
//-----------------------------------------------------------------------------
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0]; 
//-----------------------------------------------------------------------------
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
//-----------------------------------------------------------------------------
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 
//-----------------------------------------------------------------------------
#define FINDMINMAX(x0,x1,x2,min,max) \
  min = max = x0;   \
  if(x1<min) min=x1;\
  if(x1>max) max=x1;\
  if(x2<min) min=x2;\
  if(x2>max) max=x2;
//-----------------------------------------------------------------------------
int planeBoxOverlap(float normal[3],float d, float maxbox[3]);
int planeBoxOverlap(float normal[3],float d, float maxbox[3])
{
    int q;
    float vmin[3],vmax[3];
    for(q=kX; q<=kZ; q++)
    {   if(normal[q] > 0.0f)
        {   vmin[q]=-maxbox[q];
            vmax[q]= maxbox[q];
        } else
        {   vmin[q]= maxbox[q];
            vmax[q]=-maxbox[q];
        }
    }

    if(DOT(normal,vmin)+d> 0.0f) return 0;
    if(DOT(normal,vmax)+d>=0.0f) return 1;
    return 0;
}

//-----------------------------------------------------------------------------
#define AXISTEST_X01(a, b, fa, fb)                           \
        p0 = a*v0[kY] - b*v0[kZ];                            \
        p2 = a*v2[kY] - b*v2[kZ];                            \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;}   \
        rad = fa * boxhalfsize[kY] + fb * boxhalfsize[kZ];   \
        if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)                            \
        p0 = a*v0[kY] - b*v0[kZ];                            \
        p1 = a*v1[kY] - b*v1[kZ];                            \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}   \
        rad = fa * boxhalfsize[kY] + fb * boxhalfsize[kZ];   \
        if(min>rad || max<-rad) return 0;
//-----------------------------------------------------------------------------
#define AXISTEST_Y02(a, b, fa, fb)                           \
        p0 = -a*v0[kX] + b*v0[kZ];                           \
        p2 = -a*v2[kX] + b*v2[kZ];                           \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;}   \
        rad = fa * boxhalfsize[kX] + fb * boxhalfsize[kZ];   \
        if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)                            \
        p0 = -a*v0[kX] + b*v0[kZ];                           \
        p1 = -a*v1[kX] + b*v1[kZ];                           \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}   \
        rad = fa * boxhalfsize[kX] + fb * boxhalfsize[kZ];   \
        if(min>rad || max<-rad) return 0;
//-----------------------------------------------------------------------------
#define AXISTEST_Z12(a, b, fa, fb)                           \
        p1 = a*v1[kX] - b*v1[kY];                            \
        p2 = a*v2[kX] - b*v2[kY];                            \
        if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;}   \
        rad = fa * boxhalfsize[kX] + fb * boxhalfsize[kY];   \
        if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)                            \
        p0 = a*v0[kX] - b*v0[kY];                            \
        p1 = a*v1[kX] - b*v1[kY];                            \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}   \
        rad = fa * boxhalfsize[kX] + fb * boxhalfsize[kY];   \
        if(min>rad || max<-rad) return 0;

//-----------------------------------------------------------------------------
int triBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3]);
int triBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3])
{
    // use separating axis theorem to test overlap between triangle and box
    // need to test for overlap in these directions:
    // 1) the {x,y,z}-directions (actually, since we use the AABB of the triangle
    //    we do not even need to test these)
    // 2) normal of the triangle
    // 3) crossproduct(edge from tri, {x,y,z}-directin)
    //    this gives 3x3=9 more tests
    float v0[3],v1[3],v2[3];
    float min,max,d,p0,p1,p2,rad,fex,fey,fez;
    float normal[3],e0[3],e1[3],e2[3];

    // 1) first test overlap in the {x,y,z}-directions
    // find min, max of the triangle each direction, and test for overlap in
    // that direction -- this is equivalent to testing a minimal AABB around
    // the triangle against the AABB
#if 1
    // This is the fastest branch on Sun
    // move everything so that the boxcenter is in (0,0,0)
    SUB(v0,triverts[0],boxcenter);
    SUB(v1,triverts[1],boxcenter);
    SUB(v2,triverts[2],boxcenter);

    // test in kX-direction
    FINDMINMAX(v0[kX],v1[kX],v2[kX],min,max);
    if(min>boxhalfsize[kX] || max<-boxhalfsize[kX]) return 0;

    // test in kY-direction
    FINDMINMAX(v0[kY],v1[kY],v2[kY],min,max);
    if(min>boxhalfsize[kY] || max<-boxhalfsize[kY]) return 0;

    // test in kZ-direction
    FINDMINMAX(v0[kZ],v1[kZ],v2[kZ],min,max);
    if(min>boxhalfsize[kZ] || max<-boxhalfsize[kZ]) return 0;
#else
    // another implementation
    // test in kX
    v0[kX]=triverts[0][kX]-boxcenter[kX];
    v1[kX]=triverts[1][kX]-boxcenter[kX];
    v2[kX]=triverts[2][kX]-boxcenter[kX];
    FINDMINMAX(v0[kX],v1[kX],v2[kX],min,max);
    if(min>boxhalfsize[kX] || max<-boxhalfsize[kX]) return 0;

    // test in kY
    v0[kY]=triverts[0][kY]-boxcenter[kY];
    v1[kY]=triverts[1][kY]-boxcenter[kY];
    v2[kY]=triverts[2][kY]-boxcenter[kY];
    FINDMINMAX(v0[kY],v1[kY],v2[kY],min,max);
    if(min>boxhalfsize[kY] || max<-boxhalfsize[kY]) return 0;

    // test in kZ
    v0[kZ]=triverts[0][kZ]-boxcenter[kZ];
    v1[kZ]=triverts[1][kZ]-boxcenter[kZ];
    v2[kZ]=triverts[2][kZ]-boxcenter[kZ];
    FINDMINMAX(v0[kZ],v1[kZ],v2[kZ],min,max);
    if(min>boxhalfsize[kZ] || max<-boxhalfsize[kZ]) return 0;
#endif

    // 2)
    // test if the box intersects the plane of the triangle
    // compute plane equation of triangle: normal*x+d=0
    SUB(e0,v1,v0);      // tri edge 0
    SUB(e1,v2,v1);      // tri edge 1
    CROSS(normal,e0,e1);
    d=-DOT(normal,v0);  // plane eq: normal.x+d=0

    if(!planeBoxOverlap(normal,d,boxhalfsize)) return 0;

    // compute the last triangle edge
    SUB(e2,v0,v2);

    // 3)
    fex = fabs(e0[kX]);
    fey = fabs(e0[kY]);
    fez = fabs(e0[kZ]);
    AXISTEST_X01(e0[kZ], e0[kY], fez, fey);
    AXISTEST_Y02(e0[kZ], e0[kX], fez, fex);
    AXISTEST_Z12(e0[kY], e0[kX], fey, fex);

    fex = fabs(e1[kX]);
    fey = fabs(e1[kY]);
    fez = fabs(e1[kZ]);
    AXISTEST_X01(e1[kZ], e1[kY], fez, fey);
    AXISTEST_Y02(e1[kZ], e1[kX], fez, fex);
    AXISTEST_Z0(e1[kY], e1[kX], fey, fex);

    fex = fabs(e2[kX]);
    fey = fabs(e2[kY]);
    fez = fabs(e2[kZ]);
    AXISTEST_X2(e2[kZ], e2[kY], fez, fey);
    AXISTEST_Y1(e2[kZ], e2[kX], fez, fex);
    AXISTEST_Z12(e2[kY], e2[kX], fey, fex);

    return 1;
}
//-----------------------------------------------------------------------------
int triBoxBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3]);
int triBoxBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3])
{
    // use separating axis theorem to test overlap between triangle and box
    // need to test for overlap in these directions:
    // 1) the {x,y,z}-directions (actually, since we use the AABB of the triangle
    //    we do not even need to test these)
    // 2) normal of the triangle
    // 3) crossproduct(edge from tri, {x,y,z}-directin)
    //    this gives 3x3=9 more tests
    float v0[3],v1[3],v2[3];
    float min,max;

    // 1) first test overlap in the {x,y,z}-directions
    // find min, max of the triangle each direction, and test for overlap in
    // that direction -- this is equivalent to testing a minimal AABB around
    // the triangle against the AABB
   
    // This is the fastest branch on Sun
    // move everything so that the boxcenter is in (0,0,0)
    SUB(v0,triverts[0],boxcenter);
    SUB(v1,triverts[1],boxcenter);
    SUB(v2,triverts[2],boxcenter);

    // test in kX-direction
    FINDMINMAX(v0[kX],v1[kX],v2[kX],min,max);
    if(min>boxhalfsize[kX] || max<-boxhalfsize[kX]) return 0;

    // test in kY-direction
    FINDMINMAX(v0[kY],v1[kY],v2[kY],min,max);
    if(min>boxhalfsize[kY] || max<-boxhalfsize[kY]) return 0;

    // test in kZ-direction
    FINDMINMAX(v0[kZ],v1[kZ],v2[kZ],min,max);
    if(min>boxhalfsize[kZ] || max<-boxhalfsize[kZ]) return 0;

    return 1;
}
//-----------------------------------------------------------------------------
#endif