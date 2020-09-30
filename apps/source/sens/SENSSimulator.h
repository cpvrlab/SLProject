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
    virtual ~SENSSimulated() {}

    bool isThreadRunning() const { return _threadIsRunning; }

protected:
    using StartSimCB         = std::function<SENSTimePt(void)>;
    using SensorSimStoppedCB = std::function<void(void)>;

    SENSSimulated(StartSimCB                              startSimCB,
                  SensorSimStoppedCB                      sensorSimStoppedCB,
                  std::vector<std::pair<SENSTimePt, T>>&& data)
      : _startSimCB(startSimCB),
        _sensorSimStoppedCB(sensorSimStoppedCB),
        _data(data)
    {
    }

    virtual void feedSensorData(const int counter) = 0;

    void startSim()
    {
        const SENSTimePt& startTimePt = _startSimCB();
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

    const SENSTimePt& firstTimePt()
    {
        assert(_data.size());
        return _data.front().first;
    }

    void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt)
    {
        _commonSimStartTimePt = commonSimStartTimePt;
    }

private: //methods
    void feedSensor(const SENSTimePt startTimePt, const SENSTimePt simStartTimePt)
    {
        _threadIsRunning = true;
        int counter      = 0;

        auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
        SENSTimePt simTime       = simStartTimePt + passedSimTime;
        //bring counter close to startTimePt (maybe we skip some values to synchronize simulated sensors)
        while (counter < _data.size() && _data[counter].first < simTime)
            counter++;

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
                SENS_DEBUG("feed sensor with latency: %d us", std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count());
                feedSensorData(counter);

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

            //simTime should now be smaller than valueTime because valueTime is in the simulation future
            SENSMicroseconds waitTimeUs = std::chrono::duration_cast<SENSMicroseconds>(valueTime - simTime);
            //We reduce the wait time because thread sleep is not very exact (best is not to wait at all)
            SENSMicroseconds reducedWaitTimeUs((long)(0.1 * (double)waitTimeUs.count()));

            std::unique_lock<std::mutex> lock(_mutex);
            _condVar.wait_for(lock, reducedWaitTimeUs, [&] { return _stop == true; });
            //SENS_DEBUG("wait time %d us", reducedWaitTimeUs.count());

            if (_stop)
                break;
        }

        _threadIsRunning = false;
    }

protected:
    std::vector<std::pair<SENSTimePt, T>> _data;
    //inform simulator that sensor was stopped
    SensorSimStoppedCB _sensorSimStoppedCB;

private:
    std::thread             _thread;
    std::condition_variable _condVar;
    std::mutex              _mutex;
    bool                    _stop = false;

    StartSimCB _startSimCB;

    SENSTimePt _commonSimStartTimePt;

    bool _threadIsRunning = false;
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
            startSim();
            _running = true;
        }

        return _running;
    }

    void stop() override
    {
        if (_running)
        {
            stopSim();
            _running = false;
            //callback here and not in base class because stopSim is also called in startSim
            _sensorSimStoppedCB();
        }
    }

private:
    //only SENSSimulator can instantiate
    SENSSimulatedGps(StartSimCB                                              startSimCB,
                     SensorSimStoppedCB                                      sensorSimStoppedCB,
                     std::vector<std::pair<SENSTimePt, SENSGps::Location>>&& data)
      : SENSSimulated(startSimCB, sensorSimStoppedCB, std::move(data))
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
            startSim();
            _running = true;
        }

        return _running;
    }

    void stop() override
    {
        if (_running)
        {
            stopSim();
            _running = false;
            //callback here and not in base class because stopSim is also called in startSim
            _sensorSimStoppedCB();
        }
    }

private:
    SENSSimulatedOrientation(StartSimCB                                                  startSimCB,
                             SensorSimStoppedCB                                          sensorSimStoppedCB,
                             std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>&& data)
      : SENSSimulated(startSimCB, sensorSimStoppedCB, std::move(data))
    {
    }

    void feedSensorData(const int counter) override
    {
        setOrientation(_data[counter].second);
    }
};

class SENSSimulatedCamera : public SENSCameraBase
  , public SENSSimulated<int>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedCamera()
    {
        stop();
    }

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  cv::Size                      imgRGBSize           = cv::Size(),
                                  bool                          mirrorV              = false,
                                  bool                          mirrorH              = false,
                                  bool                          convToGrayToImgManip = false,
                                  int                           imgManipWidth        = -1,
                                  bool                          provideIntrinsics    = true,
                                  float                         fovDegFallbackGuess  = 65.f) override
    {
        if (_started)
        {
            Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
            return _config;
        }

        cv::Size targetSize;
        if (imgRGBSize.width > 0 && imgRGBSize.height > 0)
        {
            targetSize.width  = imgRGBSize.width;
            targetSize.height = imgRGBSize.height;
        }
        else
        {
            targetSize.width  = streamConfig.widthPix;
            targetSize.height = streamConfig.heightPix;
        }

        cv::Size imgManipSize(imgManipWidth,
                              (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width));

        //retrieve all camera characteristics
        if (_captureProperties.size() == 0)
            captureProperties();

        //start the sensor simulation
        startSim();

        return _config;
    }

    void stop() override
    {
        if (_started)
        {
            stopSim();
            _started = false;
            //callback here and not in base class because stopSim is also called in startSim
            _sensorSimStoppedCB();
        }
    }

    SENSFramePtr latestFrame() override
    {
        SENSFramePtr newFrame;

        {
            //lock and assign
        }

        return newFrame;
    }

    const SENSCaptureProperties& captureProperties() override
    {
        if (!_captureProperties.size())
        {
            SENSCameraDeviceProperties characteristics("0", SENSCameraFacing::UNKNOWN);
            //find out capture size
            //cv::VideoCapture cap;
            //cap.open(_output)
            //characteristics.add(newSize.width, newSize.height, -1.f);
            _captureProperties.push_back(characteristics);
        }

        return _captureProperties;
    }

private:
    SENSSimulatedCamera(StartSimCB                                startSimCB,
                        SensorSimStoppedCB                        sensorSimStoppedCB,
                        std::vector<std::pair<SENSTimePt, int>>&& data,
                        std::string                               videoFileName)
      : SENSSimulated(startSimCB, sensorSimStoppedCB, std::move(data)),
        _videoFileName(videoFileName)
    {
        _permissionGranted = true;
    }

    void feedSensorData(const int counter) override
    {
        //do the post processing and assign new SENSFramePtr
    }

    std::string _videoFileName;
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
                loadGpsData(dirName);
                loadOrientationData(dirName);
                loadCameraData(dirName);

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
    SENSSimulatedCamera*      getCameraSensorPtr() { return _camera.get(); }

private:
    SENSTimePt onStart()
    {
        if (!_running)
        {
            _startTimePoint = SENSClock::now();
            _running        = true;
        }

        return _startTimePoint;
    }

    void onSensorSimStopped()
    {
        //if no sensor is running anymore, we stop the simulation
        if (!_orientation->isThreadRunning() &&
            !_gps->isThreadRunning() &&
            !_camera->isThreadRunning())
        {
            _running = false;
        }
    }

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

        if (_camera)
        {
            const SENSTimePt& tp = _camera->firstTimePt();
            if (!initialized)
            {
                initialized     = true;
                _simStartTimePt = tp;
            }
            else if (tp < _simStartTimePt)
                _simStartTimePt = tp;
        }

        if (_gps)
            _gps->setCommonSimStartTimePt(_simStartTimePt);
        if (_orientation)
            _orientation->setCommonSimStartTimePt(_simStartTimePt);
        if (_camera)
            _camera->setCommonSimStartTimePt(_simStartTimePt);
    }

    void loadGpsData(const std::string& dirName)
    {
        std::string gpsFileName = dirName + "gps.txt";
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
                      new SENSSimulatedGps(std::bind(&SENSSimulator::onStart, this),
                                           std::bind(&SENSSimulator::onSensorSimStopped, this),
                                           std::move(data)));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", gpsFileName.c_str());
        }
    }

    void loadOrientationData(const std::string& dirName)
    {
        std::string orientationFileName = dirName + "orientation.txt";
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
                      new SENSSimulatedOrientation(std::bind(&SENSSimulator::onStart, this),
                                                   std::bind(&SENSSimulator::onSensorSimStopped, this),
                                                   std::move(data)));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", orientationFileName.c_str());
        }
    }

    void loadCameraData(const std::string& dirName)
    {
        std::string textFileName  = dirName + "camera.txt";
        std::string videoFileName = dirName + "video.mp4";

        //check if directory contains gps.txt
        if (Utils::fileExists(textFileName) && Utils::fileExists(videoFileName))
        {
            std::vector<std::pair<SENSTimePt, int>> data;
            std::string                             line;
            ifstream                                file(textFileName);
            if (file.is_open())
            {
                while (std::getline(file, line))
                {
                    //cout << line << '\n';
                    long                     readTimePt;
                    int                      frameIndex;
                    std::vector<std::string> values;
                    Utils::splitString(line, ' ', values);
                    if (values.size() == 5)
                    {
                        readTimePt = std::stof(values[0]);
                        frameIndex = std::stof(values[1]);

                        SENSMicroseconds readTimePtUs(readTimePt);
                        SENSTimePt       tPt(readTimePtUs);
                        data.push_back(std::make_pair(tPt, frameIndex));
                    }
                }
                file.close();

                if (data.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _camera = std::unique_ptr<SENSSimulatedCamera>(
                      new SENSSimulatedCamera(std::bind(&SENSSimulator::onStart, this),
                                              std::bind(&SENSSimulator::onSensorSimStopped, this),
                                              std::move(data),
                                              videoFileName));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s or ", textFileName.c_str(), videoFileName.c_str());
        }
    }

    std::unique_ptr<SENSSimulatedGps>         _gps;
    std::unique_ptr<SENSSimulatedOrientation> _orientation;
    std::unique_ptr<SENSSimulatedCamera>      _camera;

    //real start time point (when SENSSimulation::start was called)
    SENSTimePt _startTimePoint;
    //start time point of simulation
    SENSTimePt _simStartTimePt;

    std::atomic_bool _running{false};
};

#endif
