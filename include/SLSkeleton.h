//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLSKELETON_H
#define SLSKELETON_H


#include <stdafx.h>
#include <SLBone.h>

class SLSkeleton
{
public:
    void update();

    SLBone* getBone(SLuint handle);
    SLint numBones() const { return _boneMap.size(); }
    void getBoneWorldMatrices(SLMat4f* boneWM);



protected:
    SLBone* _root;
    std::map<SLuint, SLBone*> _boneMap; //!< bone map for fast acces of bones
};

#endif