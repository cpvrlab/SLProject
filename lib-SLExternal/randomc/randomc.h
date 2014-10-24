/*!	\file photonmap.h

	This file is part of the renderBitch distribution.
	Copyright (C) 2002 Wojciech Jarosz

	Contact:
		Wojciech Jarosz - renderBitch@renderedRealities.net
		See http://renderedRealities.net/ for more information.

	Original code is by Agner Fog. Modified by Wojciech Jarosz to include
	a quasi-random sobol sequence generator (TRandomSobol, in this file,
	and sobol.cpp). All other files from the random number library are
	unchanged.

	==========================================

	This file contains class declarations for the C++ library of uniform
	random number generators.

	Overview of classes:
	====================

	class TRanrotBGenerator:
	Random number generator of type RANROT-B.
	Source file ranrotb.cpp

	class TRanrotWGenerator:
	Random number generator of type RANROT-W.
	Source file ranrotw.cpp

	class TRandomMotherOfAll:
	Random number generator of type Mother-of-All (Multiply with carry).
	Source file mother.cpp

	class TRandomMersenne:
	Random number generator of type Mersenne twister.
	Source file mersenne.cpp

	class TRandomMotRot:
	Combination of Mother-of-All and RANROT-W generators.
	Source file ranmoro.cpp and motrot.asm.
	Coded in assembly language for improved speed.
	Must link in RANDOMAO.LIB or RANDOMAC.LIB.


	Member functions (methods):
	===========================

	All these classes have identical member functions:

	Constructor(long int seed):
	The seed can be any integer. Usually the time is used as seed.
	Executing a program twice with the same seed will give the same sequence of
	random numbers. A different seed will give a different sequence.

	double Random();
	Gives a floating point random number in the interval 0 <= x < 1.
	The resolution is 32 bits in TRanrotBGenerator, TRandomMotherOfAll and
	TRandomMersenne. 52 or 63 bits in TRanrotWGenerator. 63 bits in 
	TRandomMotRot.

	int IRandom(int min, int max);
	Gives an integer random number in the interval min <= x <= max.
	The resolution is the same as for Random().

	unsigned long BRandom();
	Gives 32 random bits. 
	Only available in the classes TRanrotWGenerator and TRandomMersenne.


	Example:
	========
	The file EX-RAN.CPP contains an example of how to generate random numbers.


	Further documentation:
	======================
	The file randomc.htm contains further documentation on these random number
	generators.

*/

#ifndef RANDOMC_H
#define RANDOMC_H

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
class TRanrotBGenerator
{	// encapsulate random number generator

	enum constants
	{					// define parameters
		KK = 17,
		JJ = 10,
		R1 = 13,
		R2 =  9
	};

public:
	void RandomInit(long int seed); 	// initialization
	int IRandom(int min, int max);		// get integer random number in desired interval
	double Random();					// get floating point random number
	TRanrotBGenerator(long int seed);	// constructor

protected:
	int p1, p2; 						// indexes into buffer
	unsigned long randbuffer[KK];		// history buffer

#ifdef SELF_TEST
	unsigned long randbufcopy[KK*2];	// used for self-test
#endif

};

#ifdef HIGH_RESOLUTION
	typedef long double trfloat;		// use long double precision
#else
	typedef double trfloat; 			// use double precision
#endif

class TRanrotWGenerator
{	// encapsulate random number generator

	enum constants	// define parameters
	{
		KK = 17,
		JJ = 10,
		R1 = 19,
		R2 =  27
	};

public:
	void RandomInit(long int seed);			// initialization
	int IRandom(int min, int max);			// get integer random number in desired interval
	trfloat Random();						// get floating point random number
	unsigned long BRandom();				// output random bits  
	TRanrotWGenerator(long int seed);		// constructor

protected:
	int p1, p2;								// indexes into buffer
	union									// used for conversion to float
	{		
		trfloat randp1;
		unsigned long randbits[3];
	};  

	unsigned long randbuffer[KK][2];		// history buffer

#ifdef SELF_TEST
		unsigned long randbufcopy[KK*2][2];	// used for self-test
#endif

};

class TRandomMotherOfAll
{	// encapsulate random number generator

public:
	void RandomInit(long int seed);			// initialization
	int IRandom(int min, int max);			// get integer random number in desired interval
	double Random();						// get floating point random number
	TRandomMotherOfAll(long int seed);		// constructor

protected:
	double x[5];							// history buffer
};

class TRandomMersenne
{	// encapsulate random number generator

	enum constants
	{   
#if 1
		// define constants for MT11213A
		N = 351, M = 175, R = 19, THE_MATRIX_A = 0xEABD75F5,
		TEMU = 11, TEMS = 7, TEMT = 15, TEML = 17, TEMB = 0x655E5280, TEMC = 0xFFD58000
#else	 
		// or constants for MT19937
		N = 624, M = 397, R = 31, THE_MATRIX_A = 0x9908B0DF,
		TEMU = 11, TEMS = 7, TEMT = 15, TEML = 18, TEMB = 0x9D2C5680, TEMC = 0xEFC60000
#endif
	};

public:
	TRandomMersenne(long int seed) { RandomInit(seed); }
	void RandomInit(long int seed);				// re-seed
	long IRandom(long min, long max);			// output random integer
	double Random();							// output random float
	unsigned long BRandom();					// output random bits

private:
	unsigned long mt[N];						// state vector
	int mti;									// index into mt
};

class TRandomMotRot
{	// encapsulate random number generator from assembly library

public:
	int IRandom(int min, int max);		// get integer random number in desired interval
	float Random();						// get floating point random number
	TRandomMotRot(long int seed);		// constructor
};

class TRandomSobol
{	// encapsulate random number generator

	enum constants
	{
		MAXBITS = 30,
		MAXDIMS = 6
	};

public:
	TRandomSobol(void)
	{	// constructor
		mdeg[0] = 0;
		mdeg[1] = 1;
		mdeg[2] = 2;
		mdeg[3] = 3;
		mdeg[4] = 3;
		mdeg[5] = 4;
		mdeg[6] = 4;

		ip[0] = 0;
		ip[1] = 0;
		ip[2] = 1;
		ip[3] = 1;
		ip[4] = 2;
		ip[5] = 1;
		ip[6] = 4;

		iv[0]	= 0;
		iv[1]	= 1;
		iv[2]	= 1;
		iv[3]	= 1;
		iv[4]	= 1;
		iv[5]	= 1;
		iv[6]	= 1;
		iv[7]	= 3;
		iv[8]	= 1;
		iv[9]	= 3;
		iv[10]	= 3;
		iv[11]	= 1;
		iv[12]	= 1;
		iv[13]	= 5;
		iv[14]	= 7;
		iv[15]	= 7;
		iv[16]	= 3;
		iv[17]	= 3;
		iv[18]	= 5;
		iv[19]	= 15;
		iv[20]	= 11;
		iv[21]	= 5;
		iv[22]	= 15;
		iv[23]	= 13;
		iv[23]	= 9;

		RandomInit();
	}

	void RandomInit(void);					// re-seed
	void Random(float x[], const int& n);	// output random float vector

private:
	float fac;
	unsigned long in,ix[MAXDIMS+1],*iu[MAXBITS+1];
	unsigned long mdeg[MAXDIMS+1];
	unsigned long ip[MAXDIMS+1];
	unsigned long iv[MAXDIMS*MAXBITS+1];
};
  
#endif
