#ifndef SENS_ORIENTATION_H
#define SENS_ORIENTATION_H

#include <mutex>

class SENSOrientation
{
public:
    struct Quat
	{
		float quatX = 0.f;
		float quatY = 0.f; 
		float quatZ = 0.f; 
		float quatW = 0.f;
	};
	
    virtual ~SENSOrientation() {}
    //start gps sensor
    virtual bool start() = 0;
    virtual void stop()  = 0;

    Quat getOrientation();

    bool isRunning() { return _running; }

protected:
    void setOrientation(Quat orientation);

    bool _running = false;

private:
    std::mutex _orientationMutex;

    Quat _orientation;
};

#endif
