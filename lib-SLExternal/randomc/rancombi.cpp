/************************ RANCOMBI.CPP ****************** AgF 2001-10-18 *
*                                                                        *
*  This file defines a template class for combining two different        *
*  random number generators. A combination of two random number          *
*  generators is generally better than any of the two alone.             *
*  The two generators should preferably be of very different design.     *
*                                                                        *
*  Instructions:                                                         *
*  To make a combined random number generator, insert the class names    *
*  of any two random number generators, as shown in the example below.   *
*                                                                        *
*  Note: Microsoft Visual C++ 6.0 reports a syntax error in this file,   *
*  though there is none. A workaround is to replace R1 and R2 with the   *
*  actual classnames or #define's and remove the template line.          *
*                                                                        *
*************************************************************************/

// This template class combines two different random number generators
// for improved randomness. R1 and R2 are any two different random number
// generator classes.
template <class R1, class R2>
class TRandomCombined : protected R1, R2 {
  public:
  TRandomCombined(long int seed = 19) {
    RandomInit(seed);};

  void RandomInit(long int seed) {    // re-seed
    R1::RandomInit(seed);
    R2::RandomInit(seed+1);}

  double Random() {
    long double r = R1::Random() + R2::Random();
    if (r >= 1.) r -= 1.;
    return r;}
    
  long IRandom(long min, long max){       // output random integer
    // get integer random number in desired interval
    int iinterval = max - min + 1;
    if (iinterval <= 0) return -1; // error
    int i = iinterval * Random(); // truncate
    if (i >= iinterval) i = iinterval-1;
    return min + i;}};

  
//////////////////////////////////////////////////////////////////////////
/* Example showing how to use the combined random number generator:
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include "randomc.h"
#include "mersenne.cpp"
#include "ranrotw.cpp"
#include "rancombine.cpp"

void main() {
  // Make an object of the template class. The names inside <> define the
  // class names of the two random number generators to combine.
  // Use time as seed.
  TRandomCombined<TRanrotWGenerator,TRandomMersenne> RG(time(0));

  for (int i=0; i<20; i++) {
    // generate 20 random floating point numbers and 20 random integers
    printf("\n%14.10f   %2i",  RG.Random(),  RG.IRandom(0,99));}

  getch();  // wait for user to press any key
  }
*/
