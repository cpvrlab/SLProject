//#############################################################################
//  File:      SLCVKeyframe.h
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVKEYFRAME_H
#define SLCVKEYFRAME_H

#include <SLCamera.h>

//-----------------------------------------------------------------------------
//! AR Keyframe node class
/*! A Keyframe is a camera with a position and additional information about key-
points that were found in this frame. It also contains descriptors for the found
keypoints.
*/
class SLCVKeyFrame : public SLCamera
{
public:


protected:


private:
};

#endif // !SLCVKEYFRAME_H
