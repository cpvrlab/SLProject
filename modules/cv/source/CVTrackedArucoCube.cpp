//#############################################################################
//  File:      CVTrackedArucoCube.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVTrackedArucoCube.h>
#include <Utils.h>
#include <Instrumentor.h>

// TODO: Replace with OpenCV classes, SL not allowed in OpenCV module
#include <SLMat4.h>
#include <Averaged.h>

#include <utility>

//-----------------------------------------------------------------------------
CVTrackedArucoCube::CVTrackedArucoCube(string calibIniPath, float edgeLength)
  : CVTrackedAruco(-1, std::move(calibIniPath)),
    _edgeLength(edgeLength),
    _averagePosition(2, CVVec3f(0.0f, 0.0f, 0.0f)),
    _averageRotation(2, SLQuat4f(0.0f, 0.0f, 0.0f, 1.0f))
{
}
//-----------------------------------------------------------------------------
bool CVTrackedArucoCube::track(CVMat          imageGray,
                               CVMat          imageRgb,
                               CVCalibration* calib)
{
    PROFILE_FUNCTION();

    /////////////////
    // TRACK FACES //
    /////////////////

    if (!trackAll(imageGray, imageRgb, calib, _roi))
    {
        return false;
    }

    ///////////////////////////////////////////////
    // COLLECT FACE TRANSFORMATIONS AND WEIGHTS  //
    ///////////////////////////////////////////////

    vector<CVVec3f>  translations;
    vector<SLQuat4f> rotations;
    vector<float>    weights;

    for (size_t i = 0; i < arucoIDs.size(); i++)
    {
        // Convert the OpenCV matrix to an SL matrix
        SLMat4f faceViewMatrix(objectViewMats[i].val);
        faceViewMatrix.transpose();

        // Calculate how much this face contributes to the total transformation
        // The steeper a face is relative to the camera,
        // the less it should contribute because the accuracy is generally poorer in this case
        float weight = faceViewMatrix.axisZ().dot(SLVec3f::AXISZ);

        // Weights below 0.3 are very inaccurate
        // If the weight is negative, this means that the z axis is pointing the wrong way
        if (weight < 0.3f)
        {
            continue;
        }

        weight = Utils::clamp(weight, 0.0f, 1.0f);

        // Move to the center of the cube
        faceViewMatrix.translate(0.0f, 0.0f, -0.5f * _edgeLength);

        // Rotate face to cube space
        SLVec3f translation = faceViewMatrix.translation();
        if (arucoIDs[i] == 0)
            ;                                            // front (no rotation)
        else if (arucoIDs[i] == 1)
            faceViewMatrix.rotate(-90, SLVec3f::AXISY);  // right
        else if (arucoIDs[i] == 2)
            faceViewMatrix.rotate(-180, SLVec3f::AXISY); // back
        else if (arucoIDs[i] == 3)
            faceViewMatrix.rotate(90, SLVec3f::AXISY);   // left
        else if (arucoIDs[i] == 4)
            faceViewMatrix.rotate(90, SLVec3f::AXISX);   // top
        else if (arucoIDs[i] == 5)
            faceViewMatrix.rotate(-90, SLVec3f::AXISX);  // bottom
        else
        {
            SL_LOG("ArUco cube: Invalid ID: %d", arucoIDs[i]);
            continue;
        }

        // Reset translation
        faceViewMatrix.m(12, translation.x);
        faceViewMatrix.m(13, translation.y);
        faceViewMatrix.m(14, translation.z);

        // Create a quaternion from the rotation of the matrix
        SLQuat4f rotation;
        rotation.fromMat3(faceViewMatrix.mat3());

        translations.emplace_back(translation.x, translation.y, translation.z);
        rotations.push_back(rotation);
        weights.push_back(weight);
    }

    //////////////////////////////////
    // AVERAGE FACE TRANSFORMATIONS //
    //////////////////////////////////

    if (!translations.empty())
    {
        // Average the translations and rotations
        CVVec3f  posNow = averageVector(translations, weights);
        SLQuat4f rotNow = averageQuaternion(rotations, weights);

        _averagePosition.set(posNow);
        _averageRotation.set(rotNow);

        CVVec3f  pos     = _averagePosition.average();
        SLQuat4f rotQuat = _averageRotation.average();
        SLMat3f  rot     = rotQuat.toMat3();

        // clang-format off
        // Convert to an OpenCV matrix
        _objectViewMat = CVMatx44f(rot.m(0), rot.m(3), rot.m(6), pos.val[0],
                                   rot.m(1), rot.m(4), rot.m(7), pos.val[1],
                                   rot.m(2), rot.m(5), rot.m(8), pos.val[2],
                                   0, 0, 0, 1);
        // clang-format on

        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
