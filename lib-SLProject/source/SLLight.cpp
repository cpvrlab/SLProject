//#############################################################################
//  File:      SLLight.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLLight.h"

//-----------------------------------------------------------------------------
SLLight::SLLight(SLfloat ambiPower,
                 SLfloat diffPower,
                 SLfloat specPower,
                 SLint   id)
{  
    // Set parameter of SLLight
    _id = id;
    _on = true;
    _spotCutoff = 180.0f;
    _spotCosCut = cos(SL_DEG2RAD*_spotCutoff);
    _spotExponent = 1.0f;

    // Set parameters of inherited SLMaterial 
    _ambient.set (ambiPower, ambiPower, ambiPower);
    _diffuse.set (diffPower, diffPower, diffPower);
    _specular.set(specPower, specPower, specPower);

    // By default there is no attenuation set. This is physically not correct
    // Default OpenGL:      kc=1, kl=0, kq=0
    // Physically correct:  kc=0, kl=0, kq=1
    // set quadratic attenuation with d = distance to light
    //                      1
    // attenuation = ------------------
    //               kc + kl*d + kq*d*d
    kc(1.0f);
    kl(0.0f);
    kq(0.0f);
}
//-----------------------------------------------------------------------------
void SLLight::kc(SLfloat kc)
{
    _kc = kc;
    _isAttenuated = (_kc==1.0f && _kl==0.0f && _kq==0.0f) ? false : true;
}
//-----------------------------------------------------------------------------
void SLLight::kl(SLfloat kl)
{
    _kl = kl;
    _isAttenuated = (_kc==1.0f && _kl==0.0f && _kq==0.0f) ? false : true;
}
//-----------------------------------------------------------------------------
void SLLight::kq(SLfloat kq)
{
    _kq = kq;
    _isAttenuated = (_kc==1.0f && _kl==0.0f && _kq==0.0f) ? false : true;
}
//-----------------------------------------------------------------------------
