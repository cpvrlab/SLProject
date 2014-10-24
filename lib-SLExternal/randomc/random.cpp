/* A C-program for MT19937: Integer version (1999/10/28)		  */
/*	genrand() generates one pseudorandom unsigned integer (32bit) */
/* which is uniformly distributed among 0 to 2^32-1  for each	  */
/* call. sgenrand(seed) sets initial values to the working area   */
/* of 624 words. Before genrand(), sgenrand(seed) must be		  */
/* called once. (seed is any 32-bit integer.)					  */
/*	 Coded by Takuji Nishimura, considering the suggestions by	  */
/* Topher Cooper and Marc Rieffel in July-Aug. 1997.			  */

/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public	   */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later		   */
/* version. 													   */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 		   */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General	   */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */ 
/* 02111-1307  USA												   */

/* Copyright (C) 1997, 1999 Makoto Matsumoto and Takuji Nishimura. */
/* Any feedback is very welcome. For any question, comments,	   */
/* see http://www.math.keio.ac.jp/matumoto/emt.html or email	   */
/* matumoto@math.keio.ac.jp 									   */

/* REFERENCE													   */
/* M. Matsumoto and T. Nishimura,								   */
/* "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform  */
/* Pseudo-Random Number Generator",                                */
/* ACM Transactions on Modeling and Computer Simulation,		   */
/* Vol. 8, No. 1, January 1998, pp 3--30.						   */

#include "random.h"

/* Initializing the array with a seed */
void sgenrand(unsigned long seed)
{
	int i;
	
	for (i = 0; i < Nq; i++)
	{
		mt[i] = seed & 0xffff0000;
		seed = 69069 * seed + 1;
		mt[i] |= (seed & 0xffff0000) >> 16;
		seed = 69069 * seed + 1;
	}
	mti = Nq;
}

/* Initialization by "sgenrand()" is an example. Theoretically, 	 */
/* there are 2^19937-1 possible states as an intial state.			 */
/* This function allows to choose any of 2^19937-1 ones.			 */
/* Essential bits in "seed_array[]" is following 19937 bits:		 */
/*	(seed_array[0]&UPPER_MASK), seed_array[1], ..., seed_array[N-1]. */
/* (seed_array[0]&LOWER_MASK) is discarded. 						 */ 
/* Theoretically,													 */
/*	(seed_array[0]&UPPER_MASK), seed_array[1], ..., seed_array[N-1]  */
/* can take any values except all zeros.							 */
void lsgenrand(unsigned long seed_array[]);
void lsgenrand(unsigned long seed_array[])
/* the length of seed_array[] must be at least N */
{
	int i;
	
	for (i = 0; i < Nq; i++) 
		mt[i] = seed_array[i];
	mti=Nq;
}

unsigned long genrand()
{
	unsigned long y;
	static unsigned long mag01[2]={0x0, MATRIX_A};
	/* mag01[x] = x * MATRIX_A	for x=0,1 */
	
	if (mti >= Nq)
	{	/* generate N words at one time */

		int kk;
		
		if (mti == Nq+1)		/* if sgenrand() has not been called, */
			sgenrand(4357);		/* a default initial seed is used	*/
		
		for (kk=0;kk<Nq-Mq;kk++)
		{
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+Mq] ^ (y >> 1) ^ mag01[y & 0x1];
		}
		for (;kk<Nq-1;kk++)
		{
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+(Mq-Nq)] ^ (y >> 1) ^ mag01[y & 0x1];
		}
		y = (mt[Nq-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
		mt[Nq-1] = mt[Mq-1] ^ (y >> 1) ^ mag01[y & 0x1];
		
		mti = 0;
	}
	
	y = mt[mti++];
	y ^= TEMPERING_SHIFT_U(y);
	y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
	y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
	y ^= TEMPERING_SHIFT_L(y);
	
	return y; 
}
