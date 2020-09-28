#ifndef SENS_GPS_H
#define SENS_GPS_H

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

class SENSGpsListener;

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

    bool permissionGranted() const { return _permissionGranted; }
    
    void registerListener(SENSGpsListener* listener);
    void unregisterListener(SENSGpsListener* listener);
    
protected:
    void setLocation(SENSGps::Location location);

    bool _running = false;
    bool _permissionGranted = false;
private:
    std::mutex _llaMutex;

    Location _location;
    
    std::vector<SENSGpsListener*> _listeners;
    std::mutex _listenerMutex;
};

class SENSGpsListener
{
public:
    virtual ~SENSGpsListener() {}
    virtual void onGps(const SENSGps::Location& loc) = 0;
};

class SENSDummyGps : public SENSGps
{
public:
    SENSDummyGps()
    {
        _permissionGranted = true;
    }
    ~SENSDummyGps();

    bool start() override;
    void stop() override;

    void addDummyPos(SENSGps::Location loc)
    {
        _dummyLocs.push_back(loc);
    }

private:
    void startSimulation();
    void stopSimulation();
    void run();

    std::thread      _thread;
    std::atomic_bool _stop{false};

    std::vector<SENSGps::Location> _dummyLocs;
};

#endif
