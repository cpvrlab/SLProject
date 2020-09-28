#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>
#include <chrono>

#include <Utils.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

typedef std::chrono::high_resolution_clock             HighResClock;
typedef std::chrono::high_resolution_clock::time_point HighResTimePoint;

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
        return _startSimCB();
    }

    void stop() override
    {
        //stop simulator (stops, if no other activated sensor is running)
        _stopSimCB();
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
        return _startSimCB();
    }

    void stop() override
    {
        _stopSimCB();
    }
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
            if (Utils::dirExists(dirName))
            {
                gpsFileName  = dirName + "gps.txt";
                gpsAvailable = loadGpsData(dirName, gpsFileName);

                orientationFileName  = dirName + "orientation.txt";
                orientationAvailable = loadOrientationData(dirName, orientationFileName);
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
    bool loadGpsData(const std::string& dirName, const std::string& gpsFileName)
    {
        bool gpsAvailable = false;
        //check if directory contains gps.txt
        if (Utils::fileExists(gpsFileName))
        {
            std::string line;
            ifstream    file(gpsFileName);
            if (file.is_open())
            {
                while (std::getline(file, line))
                {
                    //cout << line << '\n';
                    long                     readTimePt;
                    SENSGps::Location        loc;
                    std::vector<std::string> values;
                    Utils::splitString(line, ' ', values);
                    if (values.size() == 5)
                    {
                        readTimePt       = std::stof(values[0]);
                        loc.latitudeDEG  = std::stof(values[1]);
                        loc.longitudeDEG = std::stof(values[2]);
                        loc.altitudeM    = std::stof(values[3]);
                        loc.accuracyM    = std::stof(values[4]);

                        using namespace std::chrono;
                        microseconds                      readTimePtUs(readTimePt);
                        time_point<high_resolution_clock> dt(readTimePtUs);
                        _locations.push_back(std::make_pair(loc, dt));
                    }
                }
                file.close();

                gpsAvailable = true;
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", gpsFileName.c_str());
        }

        return gpsAvailable;
    }

    bool loadOrientationData(const std::string& dirName, const std::string& orientationFileName)
    {
        bool orientationAvailable = false;
        //check if directory contains orientation.txt
        if (Utils::fileExists(orientationFileName))
        {
            std::string line;
            ifstream    file(orientationFileName);
            if (file.is_open())
            {
                while (std::getline(file, line))
                {
                    //cout << line << '\n';
                    long                     readTimePt;
                    SENSOrientation::Quat    quat;
                    std::vector<std::string> values;
                    Utils::splitString(line, ' ', values);
                    if (values.size() == 5)
                    {
                        readTimePt = std::stof(values[0]);
                        quat.quatX = std::stof(values[1]);
                        quat.quatY = std::stof(values[2]);
                        quat.quatZ = std::stof(values[3]);
                        quat.quatW = std::stof(values[4]);

                        using namespace std::chrono;
                        microseconds                      readTimePtUs(readTimePt);
                        time_point<high_resolution_clock> dt(readTimePtUs);
                        _orientations.push_back(std::make_pair(quat, dt));
                    }
                }
                file.close();

                orientationAvailable = true;
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", orientationFileName.c_str());
        }

        return orientationAvailable;
    }

    std::unique_ptr<SENSSimulatedGps>         _gps;
    std::unique_ptr<SENSSimulatedOrientation> _orientation;

    int                                                             _orientationsCounter = 0;
    std::vector<std::pair<SENSOrientation::Quat, HighResTimePoint>> _orientations;

    int                                                         _locationsCounter = 0;
    std::vector<std::pair<SENSGps::Location, HighResTimePoint>> _locations;
};

#endif
