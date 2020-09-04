#ifndef SENS_NDK_GPS_H
#define SENS_NDK_GPS_H

#include <sens/SENSGps.h>
#include <android_native_app_glue.h>

class SENSNdkGps : public SENSGps
{
public:
	SENSNdkGps(JavaVM* vm, jobject* activityContext, jclass* clazz);
    void init(bool granted);

	bool start() override;
	void stop() override;

	void updateLocation(double latitudeDEG,
						double longitudeDEG,
						double altitudeM,
						float  accuracyM);
private:

    JavaVM* _vm = nullptr;
    jobject _object = nullptr;

	bool _permissionGranted = false;
};

#endif