#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>
#include <chrono>

#include <Utils.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SENSSimulator;

template<typename T>
class SENSSimulated
{
    friend SENSSimulator;
  
public:
    ~SENSSimulated() {}

protected:
    using StartSimCB = std::function<SENSTimePt(void)>;
    
    SENSSimulated(StartSimCB                              startSimCB,
                  std::vector<std::pair<SENSTimePt, T>>&& data)
      : _startSimCB(startSimCB),
        _data(data)
    {
    }

    void startSim(const SENSTimePt& startTimePt)
    {
        stopSim();

        _thread = std::thread(&SENSSimulated::feedSensor, this, startTimePt, _commonSimStartTimePt);
    }

    void stopSim()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _stop = true;
        lock.unlock();
        _condVar.notify_one();

        if (_thread.joinable())
            _thread.join();

        lock.lock();
        _stop = false;
    }

    virtual void feedSensorData(const int counter) = 0;

    void feedSensor(const SENSTimePt startTimePt, const SENSTimePt simStartTimePt)
    {
        int counter = 0;

        //bring counter close to startTimePt (maybe we skip some values to synchronize simulated sensors)
        while (counter < _data.size() && _data[counter].first < simStartTimePt)
        {
            counter++;
        }
        //end of simulation
        if (counter >= _data.size())
            return;

        while (true)
        {
            //estimate current sim time
            auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
            SENSTimePt simTime       = simStartTimePt + passedSimTime;

            //process late values
            while (counter < _data.size() && _data[counter].first < simTime)
            {
                feedSensorData(counter);
                //_gps->setLocation(_data[counter].second);
                //setting the location maybe took some time, so we update simulation time
                passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
                simTime       = simStartTimePt + passedSimTime;
                counter++;
            }

            //end of simulation
            if (counter >= _data.size())
                break;

            //locationsCounter should now point to a value in the simulation future so lets wait
            const SENSTimePt& valueTime = _data[counter].first;
            //simTime has to be smaller than valueTime
            SENSMicroseconds waitTimeMs = std::chrono::duration_cast<SENSMicroseconds>(valueTime - simTime);

            std::unique_lock<std::mutex> lock(_mutex);
            _condVar.wait_for(lock, waitTimeMs, [&] { return _stop == true; });

            if (_stop)
                break;
        }
    }

    const SENSTimePt& firstTimePt()
    {
        assert(_data.size());
        return _data.front().first;
    }
    
    void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt)
    {
        _commonSimStartTimePt = commonSimStartTimePt;
    }

    std::vector<std::pair<SENSTimePt, T>> _data;

    std::thread             _thread;
    std::condition_variable _condVar;
    std::mutex              _mutex;
    bool                    _stop = false;

    StartSimCB _startSimCB;
    SENSTimePt _commonSimStartTimePt;
};

//sensor simulator implementation
class SENSSimulatedGps : public SENSGps
  , public SENSSimulated<SENSGps::Location>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedGps()
    {
        stop();
    }
    bool start() override
    {
        if (!_running)
        {
            const SENSTimePt& startTimePt = _startSimCB();
            startSim(startTimePt);
            _running = true;
        }

        return _running;
    }

    void stop() override
    {
        if(_running)
        {
            stopSim();
            _running = false;
        }
    }

private:
    //only SENSSimulator can instantiate
    SENSSimulatedGps(StartSimCB                                              startSimCB,
                     std::vector<std::pair<SENSTimePt, SENSGps::Location>>&& data)
      : SENSSimulated(startSimCB, std::move(data))
    {
        _permissionGranted = true;
    }

    void feedSensorData(const int counter) override
    {
        setLocation(_data[counter].second);
    }
};

class SENSSimulatedOrientation : public SENSOrientation
  , public SENSSimulated<SENSOrientation::Quat>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedOrientation()
    {
        stop();
    }
    
    bool start() override
    {
        if (!_running)
        {
            const SENSTimePt& startTimePt = _startSimCB();
            startSim(startTimePt);
            _running = true;
        }

        return _running;
    }

    void stop() override
    {
        if(_running)
        {
            stopSim();
            _running = false;
        }
    }

private:
    SENSSimulatedOrientation(StartSimCB                                                  startSimCB,
                             std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>&& data)
      : SENSSimulated(startSimCB, std::move(data))
    {
    }

    void feedSensorData(const int counter) override
    {
        setOrientation(_data[counter].second);
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
        try
        {
            //check directory content and enable simulated sensors depending on this
            std::string dirName = Utils::unifySlashes(simDirName);

            if (Utils::dirExists(dirName))
            {
                std::string gpsFileName = dirName + "gps.txt";
                loadGpsData(dirName, gpsFileName);

                std::string orientationFileName = dirName + "orientation.txt";
                loadOrientationData(dirName, orientationFileName);

                //estimate common start point in record age
                estimateSimStartPoint();
            }
            else
                Utils::log("SENS", "SENSSimulator: Directory does not exist: %s", simDirName.c_str());
        }
        catch (...)
        {
            Utils::log("SENS", "SENSSimulator: Exception while parsing sensor files");
        }
    }

    SENSSimulatedGps*         getGpsSensorPtr() { return _gps.get(); }
    SENSSimulatedOrientation* getOrientationSensorPtr() { return _orientation.get(); }

    //start sensor simulators
    SENSTimePt start()
    {
        if (!_running)
        {
            _startTimePoint = SENSClock::now();
            _running        = true;
        }

        return _startTimePoint;
    }
    
    void reset()
    {
        if(_gps)
            _gps->stop();
        
        if(_orientation)
            _orientation->stop();
        
        _running = false;
    }

private:
    void estimateSimStartPoint()
    {
        //search for earliest time point
        bool initialized = false;

        if (_gps)
        {
            const SENSTimePt& tp = _gps->firstTimePt();
            if (!initialized)
            {
                initialized     = true;
                _simStartTimePt = tp;
            }
            else if (tp < _simStartTimePt)
                _simStartTimePt = tp;
        }

        if (_orientation)
        {
            const SENSTimePt& tp = _orientation->firstTimePt();
            if (!initialized)
            {
                initialized     = true;
                _simStartTimePt = tp;
            }
            else if (tp < _simStartTimePt)
                _simStartTimePt = tp;
        }
        
        if(_gps)
            _gps->setCommonSimStartTimePt(_simStartTimePt);
        if(_orientation)
            _orientation->setCommonSimStartTimePt(_simStartTimePt);
    }

    void loadGpsData(const std::string& dirName, const std::string& gpsFileName)
    {
        //check if directory contains gps.txt
        if (Utils::fileExists(gpsFileName))
        {
            std::vector<std::pair<SENSTimePt, SENSGps::Location>> data;
            std::string                                           line;
            ifstream                                              file(gpsFileName);
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
                        data.push_back(std::make_pair(tPt, loc));
                    }
                }
                file.close();

                if (data.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _gps = std::unique_ptr<SENSSimulatedGps>(
                      new SENSSimulatedGps(std::bind(&SENSSimulator::start, this),
                                           std::move(data)));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", gpsFileName.c_str());
        }
    }

    void loadOrientationData(const std::string& dirName, const std::string& orientationFileName)
    {
        //check if directory contains orientation.txt
        if (Utils::fileExists(orientationFileName))
        {
            std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>> data;
            std::string                                               line;
            ifstream                                                  file(orientationFileName);
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
                        data.push_back(std::make_pair(tPt, quat));
                    }
                }
                file.close();

                if (data.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _orientation = std::unique_ptr<SENSSimulatedOrientation>(
                      new SENSSimulatedOrientation(std::bind(&SENSSimulator::start, this),
                                                   std::move(data)));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", orientationFileName.c_str());
        }
    }

    std::unique_ptr<SENSSimulatedGps>         _gps;
    std::unique_ptr<SENSSimulatedOrientation> _orientation;

    //real start time point (when SENSSimulation::start was called)
    SENSTimePt _startTimePoint;
    //start time point of simulation
    SENSTimePt _simStartTimePt;

    std::atomic_bool _running{false};
};

#endif
