#ifndef SENS_CAMERA_H
#define SENS_CAMERA_H

#include <opencv2/core.hpp>

class SENSCamera
{
public:
	enum class Facing
	{
		FRONT,
		BACK
	};

	enum class FocusMode : int32_t {
		CONTINIOUS_AUTO_FOCUS = 0,
		FIXED_INFINITY_FOCUS
	};


	SENSCamera(SENSCamera::Facing facing)
	 : _facing(facing)
	{
	}

	virtual void start(int width, int height, FocusMode focusMode) = 0;
	virtual void stop() {};
	virtual cv::Mat getLatestFrame() = 0;
private:
	SENSCamera::Facing _facing;
};

#endif //SENS_CAMERA_H