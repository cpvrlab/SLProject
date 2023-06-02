//#############################################################################
//  File:      CVTrackedMediaPipeHands.h
//  Date:      December 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_BUILD_WITH_MEDIAPIPE
#    ifndef CVTRACKEDMEDIAPIPEHANDS_H
#        define CVTRACKEDMEDIAPIPEHANDS_H

#        include <CVTracked.h>
#        include <mediapipe.h>

//-----------------------------------------------------------------------------
//! ???
/*!
 * ??? With references to MediaPipe intro pages
 * https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
 * Online examples: https://mediapipe-studio.webapps.google.com/demo/hand_landmarker
 */
class CVTrackedMediaPipeHands : public CVTracked
{
public:
    CVTrackedMediaPipeHands(SLstring dataPath);
    ~CVTrackedMediaPipeHands();

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib) final;

private:
    void processImageInMediaPipe(CVMat imageRgb);
    void drawResults(mp_multi_face_landmark_list* landmarks,
                     CVMat                        imageRgb);

    mp_instance* _instance;
    mp_poller*   _landmarksPoller;
};
//-----------------------------------------------------------------------------

#    endif // CVTRACKEDMEDIAPIPEHANDS_H
#endif     // SL_BUILD_WITH_MEDIAPIPE
