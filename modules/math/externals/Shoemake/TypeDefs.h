/*
Type definitions used by Ken Shoemakes EulerAngles.h and Decompose.h
*/
#ifndef TYPEDEFS_H
#define TYPEDEFS_H

typedef struct {float x, y, z, w;} Quat; /* Quaternion */
enum QuatPart {X, Y, Z, W};
typedef Quat HVect; /* Homogeneous 3D vector */
typedef Quat EulerAngles;    /* (x,y,z)=ang 1,2,3, w=order code  */
typedef float HMatrix[4][4]; /* Right-handed, for column vectors */

typedef struct {
    HVect t;	/* Translation components */
    Quat  q;	/* Essential rotation	  */
    Quat  u;	/* Stretch rotation	  */
    HVect k;	/* Stretch factors	  */
    float f;	/* Sign of determinant	  */
} AffineParts;


#endif
