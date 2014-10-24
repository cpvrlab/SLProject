/************************* RANROTW.CPP ****************** AgF 1999-03-03 *
*  Random Number generator 'RANROT' type W                               *
*  This version is used when a resolution higher that 32 bits is desired.*
*                                                                        *
*  This is a lagged-Fibonacci type of random number generator with       *
*  rotation of bits.  The algorithm is:                                  *
*  Z[n] = (Y[n-j] + (Y[n-k] rotl r1)) modulo 2^(b/2)                     *
*  Y[n] = (Z[n-j] + (Z[n-k] rotl r2)) modulo 2^(b/2)                     *
*  X[n] = Y[n] + Z[n]*2^(b/2)                                            *
*                                                                        *
*  The last k values of Y and Z are stored in a circular buffer named    *
*  randbuffer.                                                           *
*  The code includes a self-test facility which will detect any          *
*  repetition of previous states.                                        *
*  The function uses a fast method for conversion to floating point.     *
*  This method relies on floating point numbers being stored in the      *
*  standard 64-bit IEEE format or the 80-bit long double format.         *
*                                                                        *
*  The theory of the RANROT type of generators and the reason for the    *
*  self-test are described at www.agner.org/random                       *
*                                                                        *
*************************************************************************/

#include "randomc.h"
#include <string.h> // some compilers require <mem.h> instead
#include <stdio.h>
#include <stdlib.h>

//#if project_builder	
// If your system doesn't have a rotate function for 32 bits integers,
// then define it thus:
static unsigned long __lrotl (unsigned long x, int r) {
  return (x << r) | (x >> (sizeof(x)*8-r));}
//#endif


// constructor:
TRanrotWGenerator::TRanrotWGenerator(long int seed) {
  RandomInit(seed);}


unsigned long TRanrotWGenerator::BRandom() {
  // generate next random number
  unsigned long y, z;
  // generate next number
  z = __lrotl(randbuffer[p1][0], R1) + randbuffer[p2][0];
  y = __lrotl(randbuffer[p1][1], R2) + randbuffer[p2][1];
  randbuffer[p1][0] = y; randbuffer[p1][1] = z;
  // rotate list pointers
  if (--p1 < 0) p1 = KK - 1;
  if (--p2 < 0) p2 = KK - 1;
#ifdef SELF_TEST
  // perform self-test
  if (randbuffer[p1][0] == randbufcopy[0][0] &&
    memcmp(randbuffer, randbufcopy[KK-p1], 2*KK*sizeof(uint32)) == 0) {
      // self-test failed
      if ((p2 + KK - p1) % KK != JJ) {
        // note: the way of printing error messages depends on system
        // In Windows you may use FatalAppExit
        printf("Random number generator not initialized");}
      else {
        printf("Random number generator returned to initial state");}
      exit(1);}
#endif
  randbits[0] = y;
  randbits[1] = z;
  return y; //!!z;  
  }


trfloat TRanrotWGenerator::Random() {
  // returns a random number between 0 and 1.
  unsigned long z = BRandom();
#ifdef HIGH_RESOLUTION            // 80 bits floats = 63 bits resolution
  randbits[1] = z | 0x80000000;
#else                             // 64 bits floats = 52 bits resolution
  randbits[1] = (z & 0x000FFFFF) | 0x3FF00000;
#endif
  return randp1 - 1.;}


int TRanrotWGenerator::IRandom(int min, int max) {
  // get integer random number in desired interval
  int iinterval = max - min + 1;
  if (iinterval <= 0) return 0x80000000;  // error
  int i = int(iinterval * Random());      // truncate
  if (i >= iinterval) i = iinterval-1;
  return min + i;}


void TRanrotWGenerator::RandomInit (long int seed) {
  // this function initializes the random number generator.
  int i, j;

  // make random numbers and put them into the buffer
  for (i=0; i<KK; i++) {
    for (j=0; j<2; j++) {
      seed = seed * 2891336453u + 1;
      randbuffer[i][j] = seed;}}
  // set exponent of randp1
  randp1 = 1.5;
#ifdef HIGH_RESOLUTION
  assert((randbits[2]&0xFFFF)==0x3FFF); // check that Intel 10-byte float format used
#else
  assert(randbits[1]==0x3FF80000); // check that IEEE double precision float format used
#endif

  // initialize pointers to circular buffer
  p1 = 0;  p2 = JJ;
#ifdef SELF_TEST
  // store state for self-test
  memcpy (randbufcopy, randbuffer, 2*KK*sizeof(uint32));
  memcpy (randbufcopy[KK], randbuffer, 2*KK*sizeof(uint32));
#endif
  // randomize some more
  for (i=0; i<31; i++) BRandom();
}

