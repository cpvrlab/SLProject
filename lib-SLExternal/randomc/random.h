/*!	\file random.h

	This file is part of the renderBitch distribution.
	Copyright (C) 2002 Wojciech Jarosz

	original code by Makoto Matsumoto and Takuji Nishimura. converted to
	C++ from C code by Wojciech Jarosz.

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 2
	of the License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

	Contact:
		Wojciech Jarosz - renderBitch@renderedRealities.net
		See http://renderedRealities.net/ for more information.
*/

#ifndef RANDOM_H
#define RANDOM_H

// Period parameters  
#define Nq 624
#define Mq 397
#define MATRIX_A 0x9908b0df		// constant vector a
#define UPPER_MASK 0x80000000	// most significant w-r bits
#define LOWER_MASK 0x7fffffff	// least significant r bits

// Tempering parameters
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned long mt[Nq];	// the array for the state vector
static int mti=Nq+1;			// mti==N+1 means mt[N] is not initialized

// initializing the array with a NONZERO seed
void sgenrand(unsigned long seed);
unsigned long genrand(void);


// RANDOM_H
#endif
