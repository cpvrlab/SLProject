//#############################################################################
//  File:      AppPenTrackingEvaluator.cpp
//  Date:      November 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <app/AppPenTrackingEvaluator.h>

#include <app/AppPenTracking.h>
#include <AppDemo.h>
#include <SLProjectScene.h>
#include <Utils.h>

#include <fstream>

//-----------------------------------------------------------------------------
/*! Starts the evaluation by resetting the state of the evaluator and adding
 * the red dot and the key event handler to the scene.
 * @param chessboardCornerSize
 */
void AppPenTrackingEvaluator::start(float chessboardCornerSize)
{
    // Reset members
    _isRunning            = true;
    _x                    = 0;
    _z                    = 0;
    _chessboardCornerSize = chessboardCornerSize;
    corners.clear();
    intCorners.clear();
    measurements.clear();

    // Add key event handler to scene
    AppDemo::scene->eventHandlers().push_back(this);

    // Create red marker sphere
    SLAssetManager* am       = AppDemo::scene->assetManager();
    SLMaterial*     material = new SLMaterial(am, "Eval Sphere Material", SLCol4f::RED, SLCol4f::BLACK, 0.0f);
    SLSphere*       mesh     = new SLSphere(am, 0.0025f, 8, 8, "Eval Sphere Mesh", material);
    _node                    = new SLNode(mesh, "Eval Sphere");
    AppDemo::scene->root3D()->addChild(_node);

    SL_LOG("Evaluation started");
}
//-----------------------------------------------------------------------------
/*! Gets the real and the measured tip position, saves them in a vector
 * and prints them out. Afterwards, the dot is moved to the next position. If there
 * are enough measurements, "finish" is called.
 */
void AppPenTrackingEvaluator::nextStep()
{
    // Calculate values
    SLVec3f corner      = currentCorner();
    SLVec3f tip         = AppPenTracking::instance().arucoPen().tipPosition();
    SLVec3f cornerToTip = tip - corner;
    float   distance    = cornerToTip.length();

    // Print values to the standard output
    SL_LOG("Offset: [%.2f, %.2f, %.2f]cm, distance: %.2fcm",
           cornerToTip.x * 100.0,
           cornerToTip.y * 100.0,
           cornerToTip.z * 100.0,
           distance * 100.0);

    // Save corners and measurement for later writing to a CSV file
    corners.push_back(corner);
    intCorners.emplace_back(_x, _z);
    measurements.push_back(tip);

    // Increment the current corner position
    _x += 2;
    if (_x > 7)
    {
        _x = 0;
        _z += 2;

        if (_z > 10)
        {
            finish();
            return;
        }
    }

    // Update the marker position
    _node->translation(currentCorner());
}
//-----------------------------------------------------------------------------
/*! Calculates the current corner position by multiplying the corner x and y
 * coordinates by the chessboard corner size
 * @return The current corner position in 3D space
 */
SLVec3f AppPenTrackingEvaluator::currentCorner()
{
    return SLVec3f((float)_x, 0.0f, (float)_z) * _chessboardCornerSize;
}
//-----------------------------------------------------------------------------
/*! The key event handler that calls "nextStep" if F7 is pressed
 */
SLbool AppPenTrackingEvaluator::onKeyPress(const SLKey key,
                                        const SLKey mod)
{
    if (key == K_F7)
    {
        nextStep();
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
/*! Removes the dot and the key event handler from the scene and writes the
 * results (actual corner position as int and float, measured position, offset,
 * distance and average distance) to a CSV file.
 */
void AppPenTrackingEvaluator::finish()
{
    // Remove the red marker sphere from the scene
    AppDemo::scene->root3D()->removeChild(_node);

    // Remove the event handler from the scene
    std::vector<SLEventHandler*>& handlers = AppDemo::scene->eventHandlers();
    handlers.erase(std::remove(handlers.begin(), handlers.end(), this), handlers.end());

    _isRunning = false;

    SL_LOG("Evaluation finished");

    // Open the CSV file
    std::string   csvFilename = "aruco-pen-evaluation.csv";
    std::ofstream csvStream(csvFilename);

    if (!csvStream.is_open())
    {
        SL_LOG("ERROR: Failed to write evaluation to \"%s\"", csvFilename.c_str());
        return;
    }

    float totalDistance = 0.0f;
    std::vector<float> distances;

    // Write the corner values to the CSV file
    // (Trust me, it just works)
    csvStream << "\"Corner\",\"Truth [cm]\",\"Measured [cm]\",\"Offset [cm]\",\"Distance [cm]\"\n";
    for (int i = 0; i < corners.size(); i++)
    {
        SLVec2<int> corner   = intCorners[i];
        SLVec3f     truth    = corners[i] * 100.0;
        SLVec3f     measured = measurements[i] * 100.0;
        SLVec3f     offset   = measured - truth;
        float       distance = offset.length();

        csvStream << "\"" << corner.x << "," << corner.y << "\",";
        csvStream << "\"" << truth.toString(", ", 1) << "\",";
        csvStream << "\"" << measured.toString(", ", 1) << "\",";
        csvStream << "\"" << offset.toString(", ", 1) << "\",";
        csvStream << "\"" << std::fixed << std::setprecision(2) << distance << "\"\n";

        totalDistance += distance;
        distances.push_back(distance);
    }

    // Write the average to the CSV file
    float avgDistance = totalDistance / (float)corners.size();
    csvStream << "\"Average\",,,,\"" << std::fixed << std::setprecision(2) << avgDistance << "\"\n";

    // Calculate the standard deviation and write it to the CSV file
    float totalDevSqr = 0;
    for(float distance : distances) {
        float dev = avgDistance - distance;
        totalDevSqr += dev * dev;
    }
    float variance = totalDevSqr / (float)corners.size();
    float stdDev = std::sqrt(variance);
    csvStream << "\"Standard Deviation\",,,,\"" << std::fixed << std::setprecision(2) << stdDev << "\"\n";

    SL_LOG("Saved evaluation to \"%s\\%s\"", Utils::getCurrentWorkingDir().c_str(), csvFilename.c_str());
}
//-----------------------------------------------------------------------------