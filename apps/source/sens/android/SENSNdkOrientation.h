#ifndef SENS_NDK_ORIENTATION_H
#define SENS_NDK_ORIENTATION_H

#include <sens/SENSOrientation.h>
#include <android_native_app_glue.h>

class SENSNdkOrientation : public SENSOrientation
{
public:
	SENSNdkOrientation(JavaVM* vm, jobject* activityContext, jclass* clazz);
    void init(bool granted);

	bool start() override;
	void stop() override;

	void updateOrientation(const Quat& orientation);
private:

    JavaVM* _vm = nullptr;
    jobject _object = nullptr;
};

#endif