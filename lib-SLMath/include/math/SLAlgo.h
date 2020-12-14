//#############################################################################
//  File:      math/SLAlgo.h
//  Author:    Michael Goettlicher
//  Purpose:   Container for general algorithm functions
//  Date:      November
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLALGO_H
#define SLALGO_H

#include <SLMath.h>
#include <SLMat3.h>
#include <SLVec3.h>

//-----------------------------------------------------------------------------
namespace SLAlgo
{
bool estimateHorizon(const SLMat3f& enuRs, const SLMat3f& sRc, SLVec3f& horizon);
};
//-----------------------------------------------------------------------------
#endif
