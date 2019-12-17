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

struct savedState
{
    float   angle;
    int32_t x;
    int32_t y;
};

struct SensorsHandler
{
    struct android_app*     _app;
    ASensorManager*         _sensorManager;
    const ASensor*          _accelerometerSensor;
    ASensorEventQueue*      _sensorEventQueue;
    struct savedState       state;
    struct SensorsCallbacks callbacks;
};

void sensorsHandler_enableAccelerometer(SensorsHandler * handler)
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

void sensorsHandler_disableAccelerometer(SensorsHandler * handler)
{
    if (handler->_accelerometerSensor != NULL)
    {
        ASensorEventQueue_disableSensor(handler->_sensorEventQueue, handler->_accelerometerSensor);
    }
}

static int32_t sensorsHandler_handle_input(struct android_app* app, AInputEvent* event)
{
    struct SensorsHandler* sensorsHandler = (struct SensorsHandler*)app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
    {
        sensorsHandler->state.x   = AMotionEvent_getX(event, 0);
        sensorsHandler->state.y   = AMotionEvent_getY(event, 0);
        return 1;
    }
    return 0;
}

static void sensorsHandler_handle_cmd(struct android_app* app, int32_t cmd)
{
    struct SensorsHandler* sensorsHandler = (struct SensorsHandler*)app->userData;
    switch (cmd)
    {
        case APP_CMD_SAVE_STATE:
            sensorsHandler->callbacks.onSaveState(sensorsHandler->callbacks.usrPtr);
            break;
        case APP_CMD_INIT_WINDOW:
            sensorsHandler->callbacks.onInit(sensorsHandler->callbacks.usrPtr, sensorsHandler->_app);
            break;
        case APP_CMD_TERM_WINDOW:
            sensorsHandler->callbacks.onClose(sensorsHandler->callbacks.usrPtr, sensorsHandler->_app);
            break;
        case APP_CMD_GAINED_FOCUS:
            sensorsHandler->callbacks.onGainedFocus(sensorsHandler->callbacks.usrPtr);
            break;
        case APP_CMD_LOST_FOCUS:
            sensorsHandler->callbacks.onLostFocus(sensorsHandler->callbacks.usrPtr);
            break;
    }
}

void initSensorsHandler(struct android_app* app, SensorsCallbacks *cb, SensorsHandler ** handlerp)
{
    SensorsHandler * handler = (SensorsHandler*)malloc (sizeof(SensorsHandler));
    memset(handler, 0, sizeof(SensorsHandler));

    *handlerp =  handler;

    app->userData     = handler;
    app->onAppCmd     = sensorsHandler_handle_cmd;
    app->onInputEvent = sensorsHandler_handle_input;

    if (app->savedState != NULL)
    {
        handler->state = *(struct savedState*)app->savedState;
    }

    handler->_app = app;
    handler->_sensorManager = ASensorManager_getInstance();//AcquireASensorManagerInstance(app);
    handler->_sensorEventQueue = ASensorManager_createEventQueue(handler->_sensorManager, app->looper, LOOPER_ID_USER, NULL, NULL);
    handler->callbacks = *cb;
}

void sensorsHandler_processEvent(SensorsHandler * handler)
{
    int                         ident;
    int                         events;
    struct android_poll_source* source;

    while ((ident = ALooper_pollAll(0, NULL, &events, (void**)&source)) >= 0)
    {
        if (source != NULL)
        {
            source->process(handler->_app, source);
        }

        if (ident == LOOPER_ID_USER)
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

        // Check if we are exiting.
        if (handler->_app->destroyRequested != 0)
        {
            handler->callbacks.onClose(handler->callbacks.usrPtr, handler->_app);
            return;
        }
    }
}
