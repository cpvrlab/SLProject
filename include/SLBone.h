//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLBONE_H
#define SLBONE_H


#include <stdafx.h>
#include <SLNode.h>

class SLBone : public SLNode
{
public:
    SLBone();
    
    void SLBone::offsetMat(const SLMat4f& mat);
    SLMat4f calculateFinalMat();    /// @todo find a better name pls

protected:
    SLuint  _handle;    //!< unique handle inside its parent skeleton
    SLMat4f _offsetMat; //!< matrix transforming this bone from bind pose to world pose
};


#endif