//#############################################################################
//  File:      AppDemoAndroidSensorQueue.h
//  Author:    Jan Dellsperger
//  Date:      Jun 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APPDEMOANDROIDSENSORQUEUE_H
#define APPDEMOANDROIDSENSORQUEUE_H

#include <stdafx.h>
#include <android/sensor.h>
#include <android/looper.h>

class AppDemoAndroidSensorQueue
{
public:
    AppDemoAndroidSensorQueue();
    void acceleration(SLfloat& x, SLfloat& y, SLfloat& z);

private:
    const ASensor* _accelerometer;
    ASensorEventQueue* _sensorEventQueue;
    const int _looperIdent = 1;
};

#endif
