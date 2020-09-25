#ifndef SENS_ORIENTATION_H
#define SENS_ORIENTATION_H

#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <Utils.h>

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

//Dummy orientation implementation with a sensor simulation backend
class SENSDummyOrientation : public SENSOrientation
{
public:
    SENSDummyOrientation()
    {
        //define one default entry
        _dummyOrientations.push_back({0, 0, 0, 0});
    }
    ~SENSDummyOrientation();

    bool start() override;
    void stop() override;

    void addDummyOrientation(SENSOrientation::Quat ori)
    {
        _dummyOrientations.push_back(ori);
    }

    //define a recorded dummy rotation
    void setupDummyOrientations();
    void readFromFile(std::string fileName);

private:
    void startSimulation();
    void stopSimulation();
    void run();

    std::thread               _thread;
    std::atomic_bool          _stop{false};
    std::chrono::milliseconds _intervalMS;

    std::vector<SENSOrientation::Quat> _dummyOrientations;
};

class SENSOrientationRecorder
{
public:
    SENSOrientationRecorder(SENSOrientation* sensor, std::string outputDir);
    ~SENSOrientationRecorder();

    bool start(std::chrono::milliseconds intervalMS);
    void stop();

private:
    void run();

    SENSOrientation* _sensor = nullptr;

    std::thread      _thread;
    std::atomic_bool _stop{false};
    std::atomic_bool _running{false};
    //recording interval in milliseconds
    std::chrono::milliseconds _intervalMS;

    std::string _outputDir;
};

#endif
