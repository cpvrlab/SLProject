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

#include <vector>
#include <SLCamera.h>

//-----------------------------------------------------------------------------
//! AR Keyframe node class
/*! A Keyframe is a camera with a position and additional information about key-
points that were found in this frame. It also contains descriptors for the found
keypoints.
*/
class SLCVKeyFrame
{
public:
    void id(int id) { _id = id; }
    void wTc(const SLCVMat& wTc) { _wTc = wTc; }

    //! get visual representation as SLPoints
    SLCamera* getSceneObject();

private:
    int _id;
    //! opencv coordinate representation: z-axis points to principlal point,
    //! x-axis to the right and y-axis down
    //! Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
    SLCVMat _wTc;

    //Pointer to visual representation object (ATTENTION: do not delete this object)
    //We do not use inheritence, because the scene is responsible for all scene objects!
    SLCamera* _camera = NULL;
};

typedef std::vector<SLCVKeyFrame> SLCVVKeyFrame;

#endif // !SLCVKEYFRAME_H
