/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <errno.h>
#include <cassert>

#include <android/sensor.h>
#include <android_native_app_glue.h>
#include <AppDemoNativeSensorsInterface.h>

#include <WAIApp.h>

#include <SLEnums.h>

struct SensorsHandler
{
    struct android_app*     _app;
    ASensorManager*         _sensorManager;
    const ASensor*          _accelerometerSensor;
    ASensorEventQueue*      _sensorEventQueue;
    struct SensorsCallbacks callbacks;
};

void sensorsHandler_enableAccelerometer(SensorsHandler* handler)
{
    if (handler->_accelerometerSensor == NULL)
    {
        handler->_accelerometerSensor = ASensorManager_getDefaultSensor(handler->_sensorManager, ASENSOR_TYPE_ACCELEROMETER);
    }

    if (handler->_accelerometerSensor == NULL)
        return;

    ASensorEventQueue_enableSensor(handler->_sensorEventQueue, handler->_accelerometerSensor);
    ASensorEventQueue_setEventRate(handler->_sensorEventQueue, handler->_accelerometerSensor, (1000L / 60) * 1000);
}

void sensorsHandler_disableAccelerometer(SensorsHandler* handler)
{
    if (handler->_accelerometerSensor != NULL)
    {
        ASensorEventQueue_disableSensor(handler->_sensorEventQueue, handler->_accelerometerSensor);
    }
}

void initSensorsHandler(struct android_app* app, SensorsCallbacks* cb, SensorsHandler** handlerp)
{
    SensorsHandler* handler = (SensorsHandler*)malloc(sizeof(SensorsHandler));
    memset(handler, 0, sizeof(SensorsHandler));

    *handlerp = handler;

    handler->_app              = app;
    handler->_sensorManager    = ASensorManager_getInstance(); //AcquireASensorManagerInstance(app);
    handler->_sensorEventQueue = ASensorManager_createEventQueue(handler->_sensorManager, app->looper, LOOPER_ID_USER, NULL, NULL);
    handler->callbacks         = *cb;
}

void sensorsHandler_processEvent(SensorsHandler* handler)
{
    if (handler->_accelerometerSensor != NULL)
    {
        ASensorEvent event;
        while (ASensorEventQueue_getEvents(handler->_sensorEventQueue, &event, 1) > 0)
        {
            handler->callbacks.onAcceleration(handler->callbacks.usrPtr, event.acceleration.x, event.acceleration.y, event.acceleration.z);
        }
    }
}
