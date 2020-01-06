#ifndef SENSORS_INTERFACE_H
#define SENSORS_INTERFACE_H

struct SensorsCallbacks
{
    void (*onAcceleration)(void* usrPtr, float x, float y, float z);

    void* usrPtr;
};

typedef struct SensorsHandler;

void sensorsHandler_enableAccelerometer(SensorsHandler* handler);

void sensorsHandler_disableAccelerometer(SensorsHandler* handler);

void initSensorsHandler(struct android_app* app, SensorsCallbacks* cb, SensorsHandler** handlerp);

void sensorsHandler_processEvent(SensorsHandler* handler);

#endif
