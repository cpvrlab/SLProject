//#############################################################################
//  File:      CVTrackedArucoCube.cpp
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <cv/CVTrackedArucoCube.h>
#include <Utils.h>
#include <SLMat4.h>

CVTrackedArucoCube::CVTrackedArucoCube(string calibIniPath, float edgeLength)
  : CVTrackedAruco(-1, calibIniPath),
    _edgeLength(edgeLength)
{
}

bool CVTrackedArucoCube::track(CVMat          imageGray,
                               CVMat          imageRgb,
                               CVCalibration* calib)
{
    if (!trackAll(imageGray, imageRgb, calib))
    {
        return false;
    }

    vector<SLVec3f>  translations;
    vector<SLQuat4f> rotations;

    for (size_t i = 0; i < arucoIDs.size(); ++i)
    {
        // Convert the OpenCV matrix to an SL matrix
        SLMat4f faceViewMatrix(objectViewMats[i].val);
        faceViewMatrix.transpose();

        // Move to the center
        faceViewMatrix.translate(0.0f, 0.0f, -0.5f * _edgeLength);

        // Rotate face to cube space
        SLVec3f translation = faceViewMatrix.translation();
        if (arucoIDs[i] == 1)
            faceViewMatrix.rotate(-90, SLVec3f::AXISY); // right
        else if (arucoIDs[i] == 2)
            faceViewMatrix.rotate(-180, SLVec3f::AXISY); // back
        else if (arucoIDs[i] == 3)
            faceViewMatrix.rotate(90, SLVec3f::AXISY); // left
        else if (arucoIDs[i] == 4)
            faceViewMatrix.rotate(90, SLVec3f::AXISX); // top
        else if (arucoIDs[i] == 5)
            faceViewMatrix.rotate(-90, SLVec3f::AXISX); // bottom

        // Reset translation
        faceViewMatrix.m(12, translation.x);
        faceViewMatrix.m(13, translation.y);
        faceViewMatrix.m(14, translation.z);

        // Convert the matrix to a translation vector and a rotation quaternion
        translations.push_back(faceViewMatrix.translation());
        SLQuat4f rotation;
        rotation.fromMat3(faceViewMatrix.mat3());
        rotations.push_back(rotation);
    }

    if (!translations.empty())
    {
        // Average the translations and rotations
        SLVec3f  pos     = averageVector(translations);
        SLQuat4f rotQuat = averageQuaternion(rotations);
        SLMat3f  rot     = rotQuat.toMat3();

        // Convert to a OpenCV matrix
        _objectViewMat = CVMatx44f(rot.m(0), rot.m(3), rot.m(6), pos.x,
                                   rot.m(1), rot.m(4), rot.m(7), pos.y,
                                   rot.m(2), rot.m(5), rot.m(8), pos.z,
                                   0, 0, 0, 1);
        return true;
    }

    return false;
}

SLVec3f CVTrackedArucoCube::averageVector(vector<SLVec3f> vectors)
{
    if (vectors.size() == 1)
        return vectors[0];

    SLVec3f total;
    for (const SLVec3f& vector : vectors)
        total += vector;
    return total / (float)vectors.size();
}

SLQuat4f CVTrackedArucoCube::averageQuaternion(vector<SLQuat4f> quaternions)
{
    if (quaternions.size() == 1)
        return quaternions[0];

    // Based on: https://math.stackexchange.com/questions/61146/averaging-quaternions

    SLQuat4f average(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < quaternions.size(); i++)
    {
        SLQuat4f quaternion = quaternions[i];
        float    weight     = 1.0f;

        if (i > 0 && quaternion.dot(quaternions[0]) < 0.0f)
            weight = -weight;

        average.set(average.x() + weight * quaternion.x(),
                    average.y() + weight * quaternion.y(),
                    average.z() + weight * quaternion.z(),
                    average.w() + weight * quaternion.w());
    }

    return average.normalized();
}