/************************* RANROTB.CPP ****************** AgF 1999-03-03 *
*  Random Number generator 'RANROT' type B                               *
*                                                                        *
*  This is a lagged-Fibonacci type of random number generator with       *
*  rotation of bits.  The algorithm is:                                  *
*  X[n] = ((X[n-j] rotl r1) + (X[n-k] rotl r2)) modulo 2^b               *
*                                                                        *
*  The last k values of X are stored in a circular buffer named          *
*  randbuffer.                                                           *
*  The code includes a self-test facility which will detect any          *
*  repetition of previous states.                                        *
*  The function uses a fast method for conversion to floating point.     *
*  This method relies on floating point numbers being stored in the      *
*  standard 64-bit IEEE format.                                          *
*                                                                        *
*  The theory of the RANROT type of generators and the reason for the    *
*  self-test are described at www.agner.org/random                       *
*                                                                        *
*************************************************************************/

#include <randomc.h>
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
TRanrotBGenerator::TRanrotBGenerator(long int seed) {
  RandomInit(seed);}


// returns a random number between 0 and 1:
double TRanrotBGenerator::Random() {
  unsigned long x;
  // generate next random number
  x = randbuffer[p1] = __lrotl(randbuffer[p2], R1) + __lrotl(randbuffer[p1], R2);
  // rotate list pointers
  if (--p1 < 0) p1 = KK - 1;
  if (--p2 < 0) p2 = KK - 1;
#ifdef SELF_TEST
  // perform self-test
  if (randbuffer[p1] == randbufcopy[0] &&
    memcmp(randbuffer, randbufcopy+KK-p1, KK*sizeof(uint32)) == 0) {
      // self-test failed
      if ((p2 + KK - p1) % KK != JJ) {
        // note: the way of printing error messages depends on system
        // In Windows you may use FatalAppExit
        printf("Random number generator not initialized");}
      else {
        printf("Random number generator returned to initial state");}
      exit(1);}
#endif
  // fast conversion to float:
  union {
    double randp1;
    unsigned long randbits[2];};
  randbits[0] = x << 20;
  randbits[1] = (x >> 12) | 0x3FF00000;
  return randp1 - 1.0;}


// returns integer random number in desired interval:
int TRanrotBGenerator::IRandom(int min, int max) 
{  int iinterval = max - min + 1;
   assert(iinterval > 0);
   int i = (int)(iinterval * Random()); // truncate
   if (i >= iinterval) i = iinterval-1;
   return min + i;
}
  

void TRanrotBGenerator::RandomInit (long int seed) {
  // this function initializes the random number generator.
  int i;
  unsigned long s = seed;

  // make random numbers and put them into the buffer
  for (i=0; i<KK; i++) {
    s = s * 2891336453u + 1;
    randbuffer[i] = s;}

  // check that the right data formats are used by compiler:
  union {
    double randp1;
    unsigned long randbits[2];};
  randp1 = 1.5;
  //assert(randbits[1]==0x3FF80000); // check that IEEE double precision float format used

  // initialize pointers to circular buffer
  p1 = 0;  p2 = JJ;
#ifdef SELF_TEST
  // store state for self-test
  memcpy (randbufcopy, randbuffer, KK*sizeof(uint32));
  memcpy (randbufcopy+KK, randbuffer, KK*sizeof(uint32));
#endif
  // randomize some more
  for (i=0; i<9; i++) Random();
}

