//#############################################################################
//  File:      SLRaySamples2D.cpp
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <chrono>
#include <assert.h>
#include <SLRaySamples2D.h>

//-----------------------------------------------------------------------------
//! Resets the sample point array by the sqrt of the no. of samples
void SLRaySamples2D::samples(SLuint x, SLuint y, SLbool evenlyDistributed)
{
    assert(x > 0 && y > 0);
    _samplesX = x;
    _samplesY = y;
    _samples  = x * y;
    _points.resize(_samples);
    if (_samples > 1) distribConcentric(evenlyDistributed);
}
//-----------------------------------------------------------------------------
/*!
Makes concentric 2D-samples points within a circle of certain radius.
With the parameter evenlyDistributed=false will the samplepoints be
denser towards the center.
*/
void SLRaySamples2D::distribConcentric(SLbool evenlyDistributed)
{
    if (_points.size())
    {
        SLfloat halfDeltaPhi = Utils::TWOPI / _samplesY * 0.5f;
        SLfloat phi, r, last_r = 1.0f;

        // Loop over radius r and angle phi
        for (SLint iR = (SLint)_samplesX - 1; iR >= 0; --iR)
        {
            r = ((SLfloat)iR) / _samplesX;
            if (evenlyDistributed) r = sqrt(r);
            r += (last_r - r) * 0.5f;

            // every 2nd circle is rotated by have delta phi for better distribution
            SLfloat iModPhi = (iR % 2) * halfDeltaPhi;
            for (SLint iPhi = (SLint)_samplesY - 1; iPhi >= 0; --iPhi)
            {
                phi = Utils::TWOPI * ((SLfloat)iPhi) / _samplesY + iModPhi;
                point((SLuint)iR, (SLuint)iPhi, SLVec2f(r * cos(phi), r * sin(phi)));
            }
            last_r = r;
        }
    }
}
//-----------------------------------------------------------------------------
/*! Concentric mapping of a x,y-position
Code taken from Peter Shirley out of "Realistic Ray Tracing"
*/
SLVec2f SLRaySamples2D::mapSquareToDisc(SLfloat x, // [0 < x <=1]
                                        SLfloat y) // [0 < y <=1]
{
    SLfloat phi, r, u, v;
    SLfloat a = 2 * x - 1;
    SLfloat b = 2 * y - 1;

    if (a > -b)
    {
        if (a > b)
        {
            r   = a;
            phi = (Utils::PI / 4) * (b / a);
        }
        else
        {
            r   = b;
            phi = (Utils::PI / 4) * (2 - a / b);
        }
    }
    else
    {
        if (a < b)
        {
            r   = -a;
            phi = (Utils::PI / 4) * (4 + b / a);
        }
        else
        {
            r = -b;
            if (b != 0)
            {
                phi = (Utils::PI / 4) * (6 - a / b);
            }
            else
                phi = 0;
        }
    }
    u = r * cos(phi);
    v = r * sin(phi);
    return SLVec2f(u, v);
}
//-----------------------------------------------------------------------------
