/*!	\file rgbe.h

	This file is part of the renderBitch distribution.
	Copyright (C) 2002 Wojciech Jarosz

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

#ifndef RGBE_INCLUDED
#define RGBE_INCLUDED

#include <SLVec3.h>

//! Definition of an rgbe class.
/*!
	Used to store irradiance values with high dynamic range in only 32 bits.
 */
class RGBE
{
public:
	unsigned char rgbe[4];

	// standard conversion from float pixels to rgbe pixels
	RGBE(const float &red = 0.0, const float &green = 0.0, const float &blue = 0.0)
	{
		setRGBE(red,green,blue);
	}

	// standard conversion from float pixels to rgbe pixels
	RGBE(const RGBE &myrgbe)
	{
		rgbe[0] = myrgbe.rgbe[0];
		rgbe[1] = myrgbe.rgbe[1];
		rgbe[2] = myrgbe.rgbe[2];
		rgbe[3] = myrgbe.rgbe[3];
	}

	// standard conversion from float pixels to rgbe pixels
	RGBE(const SLVec3f &rgb)
	{
		setRGBE(rgb.x,rgb.y,rgb.z);
	}

	inline void setRGBE(const SLVec3f &rgb)
	{
		setRGBE(rgb.x,rgb.y,rgb.z);
	}

	inline void setRGBE(const float &red, const float &green, const float &blue)
	{
		float v;
		int e;

		// v is the largest of the r g b
		v = red;
		if (green > v)	v = green;
		if (blue > v)	v = blue;

		if (v < 1e-32)
			rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
		else
		{
			float m = v;
			int n = 0;

			// get mantissa and exponent
			if (v > 1.0)
			{
				while(m > 1.0)
				{
					m *= 0.5;
					n++;
				}
			}
			else if (v <= 0.5)
			{
				while(m <= 0.5)
				{
					m *= 2.0;
					n--;
				}
			}

			e = n;

			//v = frexp(v,&e) * 256.0/v;

			v =  m * 255.0f / v;

			rgbe[0] = (unsigned char) (red * v);
			rgbe[1] = (unsigned char) (green * v);
			rgbe[2] = (unsigned char) (blue * v);
			rgbe[3] = (unsigned char) (e + 128);
		}
	}


	// standard conversion from rgbe to float pixels
	// note: Ward uses ldexp(col+0.5,exp-(128+8)).  However we wanted pixels
	//       in the range [0,1] to map back into the range [0,1].
	inline SLVec3f getRGBE() const
	{
		//SLVec3f rgb;

		// non-zero pixel
		if (rgbe[3])
		{
			float f = 1.0f;

			int e = rgbe[3]-(128+8);

			// evaluate 2^e efficiently
			if (e > 0)
			{
				for (int i = 0; i < e; i++)
				{
					f *= 2.0;
				}
			}
			else
			{
				for (int i = 0; i < -e; i++)
				{
					f /= 2.0;
				}
			}
			
			return SLVec3f((rgbe[0] + 0.5f) * f,(rgbe[1] + 0.5f) * f,(rgbe[2] + 0.5f) * f);		
		}
		else
			return SLVec3f(0.0,0.0,0.0);
	}
};


// RGBE_INCLUDED
#endif