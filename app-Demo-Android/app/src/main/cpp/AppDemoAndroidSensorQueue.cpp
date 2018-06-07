//#############################################################################
//  File:      AppDemoAndroidSensorQueue.cpp
//  Author:    Jan Dellsperger
//  Date:      Jun 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <AppDemoAndroidSensorQueue.h>

AppDemoAndroidSensorQueue::AppDemoAndroidSensorQueue()
{
    ASensorManager *sm = ASensorManager_getInstance();
    _accelerometer = ASensorManager_getDefaultSensor(sm, ASENSOR_TYPE_LINEAR_ACCELERATION);
    _sensorEventQueue = ASensorManager_createEventQueue(sm, ALooper_prepare(
            ALOOPER_PREPARE_ALLOW_NON_CALLBACKS), _looperIdent, NULL, NULL);
}

void AppDemoAndroidSensorQueue::acceleration(SLfloat &x, SLfloat& y, SLfloat& z)
{
    if (ASensorEventQueue_enableSensor(_sensorEventQueue, _accelerometer) == 0)
    {
        int ident = ALooper_pollAll(0, NULL, NULL, NULL);

        if (ident ==_looperIdent)
        {
            ASensorEvent se;
            if (ASensorEventQueue_getEvents(_sensorEventQueue, &se, 1))
            {
                x = se.acceleration.x;
                y = se.acceleration.y;
                z = se.acceleration.z;
            }
        }
    }
}
