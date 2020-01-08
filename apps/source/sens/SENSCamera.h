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

	SENSCamera(SENSCamera::Facing facing)
	 : _facing(facing)
	{
	}

	virtual void start(int width, int height) = 0;
	virtual void stop() {};
	virtual cv::Mat getLatestFrame() = 0;
private:
	SENSCamera::Facing _facing;
};

#endif //SENS_CAMERA_H