//#############################################################################
//  File:      SLLeapTool.h
//  Author:    Marc Wacker
//  Date:      January 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLEAPTOOL_H
#define SLLEAPTOOL_H

#include <stdafx.h>
#include <Leap.h>


class SLLeapTool
{
public:
    void        leapTool        (const Leap::Tool& tool) { _tool = tool; }

    SLQuat4f    toolRotation    () const;
    SLVec3f     toolTipPosition () const;
    SLVec3f     toolTipVelocity () const;

protected:
    Leap::Tool  _tool;  //!< leap tool object
};

#endif