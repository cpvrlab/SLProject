/*!	\file sobol.cpp

	This file is part of the renderBitch distribution.
	Copyright (C) 2002 Wojciech Jarosz

    Rewritten from "Numerical Recipes" C-code.

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

#include "randomc.h"
#include <iostream>

#ifndef MIN
#	define MIN(a,b)			(((a) < (b)) ? (a) : (b))
#endif

//!	Initializes the Sobol Quasi-Random number sequence.
void TRandomSobol::RandomInit(void)
{
	int j,k,l;
	unsigned long i,ipp;

	for (k = 1; k <= MAXDIMS; k++)
		ix[k] = 0;
	
	in = 0;
	if (iv[1] != 1)
		return;
	fac = 1.0/(1L << MAXBITS);
	for (j = 1,k = 0;j <= MAXBITS; j++, k+=MAXDIMS)
		iu[j] = &iv[k];

	// To allow both 1D and 2D addressing.
	for (k = 1; k <= MAXDIMS; k++)
	{
		// Stored values only require normalization.
		for (j = 1; j <= (int)mdeg[k]; j++)
			iu[j][k] <<= (MAXBITS-j);

		// Use the recurrence to get other values.
		for (j = (int)mdeg[k]+1; j <= MAXBITS; j++)
		{
			ipp=ip[k];
			i = iu[j-mdeg[k]][k];
			i ^= (i >> mdeg[k]);
			for (l = (int)mdeg[k]-1; l >= 1; l--)
			{
				if (ipp & 1) i ^= iu[j-l][k];
				ipp >>= 1;
			}
			iu[j][k] = i;
		}
	}
}



//! Returns the next number in the Quasi-Random sequence.
/*!
	When n is negative, internally initializes a set of MAXBIT direction numbers for each of MAXDIM
	different Sobol' sequences. When n is positive (but <= MAXDIM), returns as the vector x[0..n-1]
	the next values from n of these sequences. (n must not be changed between initializations.)
*/
void TRandomSobol::Random(float x[], const int& n)
{
	int j,k;
	unsigned long im;
	im = in++;
	//Find the rightmost zero bit.
	for (j = 1;j <= MAXBITS; j++)
	{
		if (!(im & 1))
			break;
		im >>= 1;
	}

	if (j > MAXBITS)
		std::cerr << "MAXBITS too small in sobseq" << std::endl;

	im = (j-1)*MAXDIMS;

	//	XOR the appropriate direction number
	//	into each component of the
	//	vector and convert to a floating
	//	number.
	//
	for (k = 1; k <= MIN(n,MAXDIMS); k++)
	{
		ix[k] ^= iv[im+k];
		x[k-1] = ix[k]*fac;
	}
}

#undef MIN
