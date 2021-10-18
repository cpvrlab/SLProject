//#############################################################################
//  File:      CVMultiTracked.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVMULTITRACKED_H
#define SLPROJECT_CVMULTITRACKED_H

#include <cv/CVTracked.h>
#include <cv/CVTypedefs.h>

#include <SLQuat4.h>

//-----------------------------------------------------------------------------
//! CVMultiTracked is used for tracking the same object in multiple frames
/*! The CVMultiTracked class averages the object view matrices of the same
 * object in multiple frames.
 * The class is given a pointer to a CVTracked which is used for tracking in
 * a single frame of data. The "track" method can then be called an arbitrary
 * number of times. Every call, the "track" method of the provided CVTracked*
 * is called and the resulting object view matrix stored in a vector.
 * To get the averaged object view matrix of the CVMultiTracked, the method
 * "combine" can called. This also clears the internal vector for new frames.
 */
class CVMultiTracked : public CVTracked
{

private:
    CVTracked* _tracked;
    CVVMatx44f _objectViewMatrices;

public:
    CVMultiTracked(CVTracked* tracked);
    ~CVMultiTracked() noexcept;

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib);

    void combine();
};
//-----------------------------------------------------------------------------
#endif //SLPROJECT_CVMULTITRACKED_H
