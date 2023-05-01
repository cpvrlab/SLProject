//#############################################################################
//  File:      CVTrackedArucoCube.h
//  Date:      September 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVTRACKEDARUCOCUBE_H
#define SLPROJECT_CVTRACKEDARUCOCUBE_H

#include <CVTypedefs.h>
#include <CVTrackedAruco.h>

// TODO: Replace with OpenCV classes, SL not allowed in OpenCV module
#include <SLQuat4.h>

//-----------------------------------------------------------------------------
class AveragedQuat4f
{
public:
    AveragedQuat4f(int numValues, SLQuat4f initValue)
    {
        init(numValues, initValue);
    }

    //! Initializes the average value array to a given value
    void init(int numValues, SLQuat4f initValue)
    {
        assert(numValues > 0 && "Num. of values must be greater than zero");
        _values.clear();
        _values.resize(numValues, initValue);
        _oneOverNumValues  = 1.0f / (float)_values.size();
        _average           = initValue;
        _currentValueIndex = 0;
    }

    //! Sets the current value in the value array and builds the average
    void set(SLQuat4f value)
    {
        assert(_values.size() > 0 && "_value vector not initialized");

        // Short cut for no averaging
        if (_values.size() == 1)
            _average = value;
        else
        {
            if (_currentValueIndex == _values.size())
                _currentValueIndex = 0;

            // Correct the sum continuously
            _values[_currentValueIndex] = value;

            _average.set(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = 0; i < _values.size(); i++)
            {
                SLQuat4f current = _values[i];
                float    weight  = _oneOverNumValues;

                if (i > 0 && current.dot(_values[0]) < -0.001)
                    weight = -weight;

                _average.set(_average.x() + current.x() * weight,
                             _average.y() + current.y() * weight,
                             _average.z() + current.z() * weight,
                             _average.w() + current.w() * weight);
            }

            _average.normalize();

            _currentValueIndex++;
        }
    }

    SLQuat4f average() { return _average; }
    size_t   size() { return _values.size(); }

private:
    float            _oneOverNumValues{};  //!< multiplier instead of divider
    vector<SLQuat4f> _values;              //!< value array
    int              _currentValueIndex{}; //!< current value index within _values
    SLQuat4f         _average;             //!< average value
};
//-----------------------------------------------------------------------------
//! OpenCV ArUco cube marker tracker class derived from CVTrackedAruco
/*! Tracks a cube of ArUco markers and averages their values. The origin
 * of the cube is in the center.
 * The markers must be placed in the following manner:
 * ID 0: front
 * ID 1: right
 * ID 2: back
 * ID 3: left
 * ID 4: top
 * ID 5: bottom
 */
class CVTrackedArucoCube : public CVTrackedAruco
{
public:
    CVTrackedArucoCube(string calibIniPath, float edgeLength);

    bool track(CVMat          imageGray,
               CVMat          imageRgb,
               CVCalibration* calib);

private:
    float _edgeLength;

    Averaged<CVVec3f> _averagePosition;
    AveragedQuat4f    _averageRotation;

public:
    CVRect _roi = CVRect(0, 0, 0, 0);
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_CVTRACKEDARUCOCUBE_H