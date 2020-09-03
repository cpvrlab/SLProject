#ifndef SENS_GPS_H
#define SENS_GPS_H

class SENSGps
{
public:
	virtual ~SENSGps() {}
	//start gps sensor
	virtual bool start() = 0;
	virtual void stop() = 0;

private:
};

#endif