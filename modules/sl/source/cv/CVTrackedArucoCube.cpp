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

// TODO: Replace with OpenCV classes, SL not allowed in OpenCV module
#include <SLMat4.h>

#include <utility>

CVTrackedArucoCube::CVTrackedArucoCube(string calibIniPath, float edgeLength)
  : CVTrackedAruco(-1, std::move(calibIniPath)),
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

    vector<CVVec3f>  translations;
    vector<SLQuat4f> rotations;
    vector<float>    weights;

    for (size_t i = 0; i < arucoIDs.size(); ++i)
    {
        // Convert the OpenCV matrix to an SL matrix
        SLMat4f faceViewMatrix(objectViewMats[i].val);
        faceViewMatrix.transpose();

        // Calculate how much this face contributes to the total transformation
        // The steeper a face is relative to the camera,
        // the less it should contribute because the accuracy is generally poorer in this case
        float weight = faceViewMatrix.axisZ().dot(SLVec3f::AXISZ);
        weight = Utils::clamp(weight, 0.0f, 1.0f);
        weights.push_back(weight);

        // Move to the center of the cube
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
        translations.emplace_back(translation.x, translation.y, translation.z);
        SLQuat4f rotation;
        rotation.fromMat3(faceViewMatrix.mat3());
        rotations.push_back(rotation);
    }

    if (!translations.empty())
    {
        // Average the translations and rotations
        CVVec3f  pos     = averageVector(translations, weights);
        SLQuat4f rotQuat = averageQuaternion(rotations, weights);
        SLMat3f  rot     = rotQuat.toMat3();

        // Convert to a OpenCV matrix
        _objectViewMat = CVMatx44f(rot.m(0), rot.m(3), rot.m(6), pos.val[0],
                                   rot.m(1), rot.m(4), rot.m(7), pos.val[1],
                                   rot.m(2), rot.m(5), rot.m(8), pos.val[2],
                                   0, 0, 0, 1);
        return true;
    }

    return false;
}

CVVec3f CVTrackedArucoCube::averageVector(vector<CVVec3f> vectors, vector<float> weights)
{
    if (vectors.size() == 1)
        return vectors[0];

    CVVec3f total;
    float totalWeights = 0.0f;

    for (int i = 0; i < vectors.size(); i++)
    {
        float weight = weights[i];
        total += vectors[i] * weight;
        totalWeights += weight;
    }

    return total / totalWeights;
}

SLQuat4f CVTrackedArucoCube::averageQuaternion(vector<SLQuat4f> quaternions, vector<float> weights)
{
    // Based on: https://math.stackexchange.com/questions/61146/averaging-quaternions

    if (quaternions.size() == 1)
        return quaternions[0];

    SLQuat4f total(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < quaternions.size(); i++)
    {
        SLQuat4f quaternion = quaternions[i];
        float    weight     = weights[i];

        if (i > 0 && quaternion.dot(quaternions[0]) < 0.0f)
            weight = -weight;

        total.set(total.x() + weight * quaternion.x(),
                  total.y() + weight * quaternion.y(),
                  total.z() + weight * quaternion.z(),
                  total.w() + weight * quaternion.w());
    }

    return total.normalized();
}