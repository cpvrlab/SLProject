#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>
#include <chrono>

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
        _permissionGranted = true;
    }

    bool start() override
    {
        if( _startSimCB())
        {
            _running = true;
        }
        return _running;
    }

    void stop() override
    {
        //stop simulator (stops, if no other activated sensor is running)
        _stopSimCB();
        _running = false;
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
        if( _startSimCB())
        {
            _running = true;
        }
        return _running;
    }

    void stop() override
    {
        _stopSimCB();
        _running = false;
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

    ~SENSSimulator()
    {
        stop();
    }

    SENSSimulatedGps*         getGpsSensorPtr() { return _gps.get(); }
    SENSSimulatedOrientation* getOrientationSensorPtr() { return _orientation.get(); }

    //start sensor simulators
    bool start()
    {
        //join running threads
        stop();

        //estimate simulation start time
        auto startTimePoint = SENSClock::now();
        
        _simStartTimePt = _locations[0].first;

        //start sensor threads for available sensors
        if (_gps)
        {
            _gpsThread = std::thread(&SENSSimulator::feedGps, this, startTimePoint, _simStartTimePt);
        }

        if (_orientation)
        {
            _orientThread = std::thread(&SENSSimulator::feedOrientation, this, startTimePoint, _simStartTimePt);
        }

        return true;
    }

    //stop sensor simulators
    void stop()
    {
        _stop = true;

        _orientCondVar.notify_one();
        _gpsCondVar.notify_one();

        if (_orientThread.joinable())
            _orientThread.join();

        if (_gpsThread.joinable())
            _gpsThread.join();

        _stop = false;
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

                        SENSMicroseconds readTimePtUs(readTimePt);
                        SENSTimePt       tPt(readTimePtUs);
                        _locations.push_back(std::make_pair(tPt, loc));
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

                        SENSMicroseconds readTimePtUs(readTimePt);
                        SENSTimePt       tPt(readTimePtUs);
                        _orientations.push_back(std::make_pair(tPt, quat));
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

    void feedGps(const SENSTimePt startTimePt, const SENSTimePt simStartTimePt)
    {
        _locationsCounter = 0;

        //bring counter close to startTimePt (maybe we skip some values to synchronize simulated sensors)
        while (_locationsCounter < _locations.size() && _locations[_locationsCounter].first < simStartTimePt)
        {
            _locationsCounter++;
        }
        //end of simulation
        if (_locationsCounter >= _locations.size())
            return;

        while (true)
        {
            //estimate current sim time
            auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
            SENSTimePt simTime       = simStartTimePt + passedSimTime;

            //process late values
            while (_locationsCounter < _locations.size() && _locations[_locationsCounter].first < simTime)
            {
                _gps->setLocation(_locations[_locationsCounter].second);
                //setting the location maybe took some time, so we update simulation time
                passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
                simTime       = simStartTimePt + passedSimTime;
                _locationsCounter++;
            }

            //end of simulation
            if (_locationsCounter >= _locations.size())
                break;

            //locationsCounter should now point to a value in the simulation future so lets wait
            const SENSTimePt& valueTime = _locations[_locationsCounter].first;
            //simTime has to be smaller than valueTime
            SENSMicroseconds waitTimeMs = std::chrono::duration_cast<SENSMicroseconds>(valueTime - simTime);

            std::unique_lock<std::mutex> lock(_gpsMutex);
            _gpsCondVar.wait_for(lock, waitTimeMs, [&] { return _stop == true; });

            if (_stop)
                break;
        }
    }

    void feedOrientation(const SENSTimePt startTimePt, const SENSTimePt simStartTimePt)
    {
    }

    std::unique_ptr<SENSSimulatedGps>         _gps;
    std::unique_ptr<SENSSimulatedOrientation> _orientation;

    int                                                       _orientationsCounter = 0;
    std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>> _orientations;

    int                                                   _locationsCounter = 0;
    std::vector<std::pair<SENSTimePt, SENSGps::Location>> _locations;

    std::thread             _orientThread;
    std::condition_variable _orientCondVar;
    std::mutex              _orientMutex;

    std::thread             _gpsThread;
    std::condition_variable _gpsCondVar;
    std::mutex              _gpsMutex;

    std::atomic_bool _stop{false};

    //start time point of simulati
    SENSTimePt _simStartTimePt;
};

#endif
