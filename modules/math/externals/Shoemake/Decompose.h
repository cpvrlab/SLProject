/**** Decompose.h - Basic declarations ****/
#include "TypeDefs.h"

#ifndef _H_Decompose
#define _H_Decompose

float polar_decomp(HMatrix M, HMatrix Q, HMatrix S);
HVect spect_decomp(HMatrix S, HMatrix U);
Quat  snuggle(Quat q, HVect *k);
void  decomp_affine(HMatrix A, AffineParts *parts);
void  invert_affine(AffineParts *parts, AffineParts *inverse);

#endif
