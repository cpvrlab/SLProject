#ifndef SENS_NDK_GPS_H
#define SENS_NDK_GPS_H

#include <sens/SENSGps.h>
#include <android_native_app_glue.h>

class SENSNdkGps : public SENSGps
{
public:
	SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz, jobject* object );
	bool start() override;
	void stop() override;

	void setPermissionGranted();
private:

    JavaVM* _vm = nullptr;
    jclass* _clazz = nullptr;
    jobject* _object = nullptr;
    jobject* _context = nullptr;

	bool _permissionGranted = false;
};

#endif