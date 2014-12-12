//#############################################################################
//  File:      SLCapsule3.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCAPSULE3_H
#define SLCAPSULE3_H

#include <stdafx.h>
#include <SLGLBuffer.h>

class SLRay;
class SLScene;

//-----------------------------------------------------------------------------
//! Unfinished capsule class, would be a faster method than using OBBs for skeletons
/*!

*/
class SLCapsule3
{
public:
    SLCapsule3();

    void build(const SLVec3f* points, SLuint numP);
    void merge(const SLCapsule3& other);

    void draw(const SLCol3f color);
    void generateVBO();

protected:
    SLVec3f     _start;
    SLVec3f     _end;
    SLfloat     _radius;
    SLGLBuffer  _buf;
};



#endif
