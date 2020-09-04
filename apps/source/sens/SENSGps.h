#ifndef SENS_GPS_H
#define SENS_GPS_H

#include <thread>
#include <mutex>
#include <atomic>

class SENSGps
{
public:
    struct Location
    {
        double latitudeDEG  = 0;
        double longitudeDEG = 0;
        double altitudeM    = 0;
        float  accuracyM    = 0.f;
    };

    virtual ~SENSGps() {}
    //start gps sensor
    virtual bool start() = 0;
    virtual void stop()  = 0;

    Location getLocation();

    bool isRunning() { return _running; }

protected:
    void setLocation(double latitudeDEG,
                     double longitudeDEG,
                     double altitudeM,
                     float  accuracyM);

    bool _running = false;

private:
    std::mutex _llaMutex;

    Location _location;
};

class SENSDummyGps : public SENSGps
{
public:
    SENSDummyGps(double latitudeDEG  = 47.142472,
                 double longitudeDEG = 7.243057,
                 double altitudeM    = 300);
    ~SENSDummyGps();

    bool start() override;
    void stop() override;

private:
    void startSimulation();
    void stopSimulation();
    void run();

    std::thread      _thread;
    std::atomic_bool _stop{false};

    SENSGps::Location _dummyLoc;
};

#endif
