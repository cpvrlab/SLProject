#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>

#include <Utils.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SENSSimulator;

class SENSSimulated
{
    friend SENSSimulator;

public:
    ~SENSSimulated() {}

protected:
    SENSSimulated(std::function<bool(void)> startSimCB,
                  std::function<void(void)> stopSimCB)
      : _startSimCB(startSimCB),
        _stopSimCB(stopSimCB)
    {
    }

protected:
    std::function<bool(void)> _startSimCB;
    std::function<void(void)> _stopSimCB;
};

//sensor simulators
class SENSSimulatedGps : public SENSGps
  , public SENSSimulated
{
    friend class SENSSimulator;

private:
    SENSSimulatedGps(std::function<bool(void)> startSimCB,
                     std::function<void(void)> stopSimCB)
      : SENSSimulated(startSimCB, stopSimCB)
    {
    }

    bool start() override
    {
        //start simulator if not already running

        return true;
    }

    void stop() override
    {
        //stop simulator (stops, if no other activated sensor is running)
    }
};

class SENSSimulatedOrientation : public SENSOrientation
  , public SENSSimulated
{
    friend class SENSSimulator;

private:
    SENSSimulatedOrientation(std::function<bool(void)> startSimCB,
                             std::function<void(void)> stopSimCB)
      : SENSSimulated(startSimCB, stopSimCB)
    {
    }

    bool start() override
    {
        return true;
    }

    void stop() override
    {
    }

    std::function<bool(void)> _startSimCB;
    std::function<void(void)> _stopSimCB;
};

class SENSSimulatedCamera //: public SENSCamera
{
public:
private:
};

//Backend simulator for sensor data recorded with SENSRecorder
class SENSSimulator
{
public:
    SENSSimulator(const std::string& simDirName)
    {
        //check directory content and enable simulated sensors depending on this
        bool gpsAvailable         = false;
        bool orientationAvailable = false;

        std::string dirName = Utils::unifySlashes(simDirName);
        std::string gpsFileName;
        std::string orientationFileName;

        try
        {
            if (Utils::dirExists(simDirName))
            {
                //check if directory contains gps.txt
                gpsFileName = dirName + "gps.txt";
                if (Utils::fileExists(gpsFileName))
                {
                    //parse it
                    
                    gpsAvailable = true;
                }
                
                //check if directory contains orientation.txt
                orientationFileName = dirName + "orientation.txt";
                if (Utils::fileExists(gpsFileName))
                {
                    //parse it
                    
                    orientationAvailable = true;
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Directory does not exist: %s", simDirName.c_str());
        }
        catch (...)
        {
            Utils::log("SENS", "SENSSimulator: Exception while parsing sensor files");
        }

        if (gpsAvailable)
        {
            //make_unique has problems with fiendship and private constructor so we use unique_ptr
            _gps = std::unique_ptr<SENSSimulatedGps>(
              new SENSSimulatedGps(std::bind(&SENSSimulator::start, this),
                                   std::bind(&SENSSimulator::stop, this)));
        }

        if (orientationAvailable)
        {
            //make_unique has problems with fiendship and private constructor so we use unique_ptr
            _orientation = std::unique_ptr<SENSSimulatedOrientation>(
              new SENSSimulatedOrientation(std::bind(&SENSSimulator::start, this),
                                           std::bind(&SENSSimulator::stop, this)));
        }
    }

    SENSSimulatedGps*         getGpsSensorPtr() { return _gps.get(); }
    SENSSimulatedOrientation* getOrientationSensorPtr() { return _orientation.get(); }

    //start sensor simulators
    bool start()
    {
        //join running threads
        stop();

        //start available sensors from file

        return true;
    }

    //stop sensor simulators
    void stop()
    {
    }

private:
    std::unique_ptr<SENSSimulatedGps>         _gps;
    std::unique_ptr<SENSSimulatedOrientation> _orientation;
};

#endif
