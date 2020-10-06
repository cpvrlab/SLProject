#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>
#include <chrono>

#include <Utils.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SENSSimulator;

//we need a common base class to be able to put it in a vector
class SENSSimulatedBase
{
    friend SENSSimulator;

public:
    virtual ~SENSSimulatedBase() {}

protected:
    virtual bool isThreadRunning() const = 0;

    virtual void feedSensorData(const int counter) = 0;
    //prepare things that may take some time (e.g. for video reading we can already read the frame and do decoding and maybe it also blocks for some reasons..)
    virtual void              prepareSensorData(const int counter){};
    virtual const SENSTimePt& firstTimePt() = 0;

    virtual void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt) = 0;
};

template<typename T>
class SENSSimulated : public SENSSimulatedBase
{
    friend SENSSimulator;

public:
    virtual ~SENSSimulated() {}

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

    void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt) override
    {
        _commonSimStartTimePt = commonSimStartTimePt;
    }

    const SENSTimePt& firstTimePt() override
    {
        assert(_data.size());
        return _data.front().first;
    }

    bool isThreadRunning() const override { return _threadIsRunning; }

private: //methods
    void feedSensor(const SENSTimePt startTimePt, const SENSTimePt simStartTimePt)
    {
        _threadIsRunning = true;
        int counter      = 0;
        
        for(int i = 0; i < _data.size(); ++i)
            SENS_DEBUG("data %d: %d", i, std::chrono::time_point_cast<SENSMicroseconds>(_data[i].first).time_since_epoch().count());

        auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
        SENSTimePt simTime       = simStartTimePt + passedSimTime;
        //bring counter close to startTimePt (maybe we skip some values to synchronize simulated sensors)
        while (counter < _data.size() && _data[counter].first < simTime)
            counter++;

        //end of simulation
        if (counter >= _data.size())
        {
            _threadIsRunning = false;
            _sensorSimStoppedCB();
            return;
        }

        while (true)
        {
            //estimate current sim time
            auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
            SENSTimePt simTime       = simStartTimePt + passedSimTime;

            //process late values
            int i = 0;
            while (counter < _data.size() && _data[counter].first < simTime)
            {
                SENS_DEBUG("feed sensor index %d with latency: %d us, SimT: %d, ValueT: %d", counter, std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count(), std::chrono::time_point_cast<SENSMicroseconds>(simTime).time_since_epoch().count(), std::chrono::time_point_cast<SENSMicroseconds>(_data[counter].first).time_since_epoch().count());
                feedSensorData(counter);

                //setting the location maybe took some time, so we update simulation time
                passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
                simTime       = simStartTimePt + passedSimTime;
                counter++;
                i++;
            }
            if (i > 0)
                SENS_DEBUG("grabbed %d frames", i);

            //end of simulation
            if (counter >= _data.size())
                break;

            //prepare things that may take some time (e.g. for video reading we can already read the frame and do decoding and maybe it also blocks for some reasons..)
            prepareSensorData(counter);

            passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
            simTime       = simStartTimePt + passedSimTime;
            
            //locationsCounter should now point to a value in the simulation future so lets wait
            const SENSTimePt& valueTime = _data[counter].first;

            //simTime should now be smaller than valueTime because valueTime is in the simulation future
            SENSMicroseconds waitTimeUs = std::chrono::duration_cast<SENSMicroseconds>(valueTime - simTime);
            //We reduce the wait time because thread sleep is not very exact (best is not to wait at all)
            SENSMicroseconds reducedWaitTimeUs((long)(0.1 * (double)waitTimeUs.count()));

            HighResTimer t;
            std::unique_lock<std::mutex> lock(_mutex);
            _condVar.wait_for(lock, reducedWaitTimeUs, [&] { return _stop == true; });
            //SENS_DEBUG("wait time %d us", reducedWaitTimeUs.count());
            //SENS_DEBUG("woke after %d us", t.elapsedTimeInMicroSec());

            if (_stop)
                break;
        }

        _threadIsRunning = false;
        _sensorSimStoppedCB();
    }

protected:
    std::vector<std::pair<SENSTimePt, T>> _data;

    //inform simulator that sensor was stopped
    SensorSimStoppedCB _sensorSimStoppedCB;
    StartSimCB         _startSimCB;

    std::thread             _thread;
    std::condition_variable _condVar;
    std::mutex              _mutex;
    bool                    _stop = false;

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
                                  cv::Size                      imgBGRSize           = cv::Size(),
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
        if (imgBGRSize.width > 0 && imgBGRSize.height > 0)
        {
            targetSize.width  = imgBGRSize.width;
            targetSize.height = imgBGRSize.height;
        }
        else
        {
            targetSize.width  = streamConfig.widthPix;
            targetSize.height = streamConfig.heightPix;
        }

        cv::Size imgManipSize(imgManipWidth,
                              (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width));

        if (!_cap.isOpened())
        {
            if (!_cap.open(_videoFileName))
                throw SENSException(SENSType::CAM, "Could not open camera simulator for filename: " + _videoFileName, __LINE__, __FILE__);
            else
            {
                //_cap.set(cv::CAP_PROP_FPS, 1);
            }
        }

        //retrieve all camera characteristics
        if (_captureProperties.size() == 0)
            captureProperties();

        //init config here
        _config = SENSCameraConfig(deviceId,
                                   streamConfig,
                                   SENSCameraFocusMode::UNKNOWN,
                                   targetSize.width,
                                   targetSize.height,
                                   imgManipSize.width,
                                   imgManipSize.height,
                                   mirrorH,
                                   mirrorV,
                                   convToGrayToImgManip);

        initCalibration(fovDegFallbackGuess);

        //start the sensor simulation
        startSim();

        _started = true;

        return _config;
    }

    void stop() override
    {
        if (_started)
        {
            stopSim();
            _started = false;
        }
    }

    SENSFramePtr latestFrame() override
    {
        cv::Mat newFrame;

        {
            std::lock_guard<std::mutex> lock(_frameMutex);
            newFrame = _frame;
            //_frame.release();
        }

        SENSFramePtr processedFrame;
        if (!newFrame.empty())
            processedFrame = postProcessNewFrame(newFrame, cv::Mat(), false);

        return processedFrame;
    }

    const SENSCaptureProperties& captureProperties() override
    {
        if (!_captureProperties.size())
        {
            SENSCameraDeviceProperties characteristics("0", SENSCameraFacing::UNKNOWN);
            //find out capture size
            if (!_cap.isOpened())
            {
                if (!_cap.open(_videoFileName))
                    throw SENSException(SENSType::CAM, "Could not open camera simulator for filename: " + _videoFileName, __LINE__, __FILE__);
                else
                {
                    //_cap.set(cv::CAP_PROP_FPS, 1);
                }
            }

            cv::Mat frame;
            _cap.read(frame);
            if (!frame.empty())
            {
                characteristics.add(frame.size().width, frame.size().height, -1.f);
                _captureProperties.push_back(characteristics);
            }
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
        if(_data[counter].second != _preparedFrameIndex)
        {
            prepareSensorData(counter);
        }

        {
            std::lock_guard<std::mutex> lock(_frameMutex);
            _frame = _preparedFrame;
        }
    }

    void prepareSensorData(const int counter) override
    {
        int frameIndex = _data[counter].second;
        if(frameIndex == _preparedFrameIndex)
            return;
        
        _cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);

        cv::Mat      frame;
        HighResTimer t;
        _cap.read(frame);
        SENS_DEBUG("read time index %d: %d", frameIndex, t.elapsedTimeInMicroSec());
        {
            _preparedFrameIndex = frameIndex;
            std::lock_guard<std::mutex> lock(_frameMutex);
            _preparedFrame = frame;
        }
    }

    std::string      _videoFileName;
    cv::VideoCapture _cap;

    cv::Mat    _preparedFrame;
    int        _preparedFrameIndex = -1;
    cv::Mat    _frame;
    std::mutex _frameMutex;
};

//Backend simulator for sensor data recorded with SENSRecorder
class SENSSimulator
{
public:
    ~SENSSimulator()
    {
        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            //_activeSensors[i]->stop();
        }
    }

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

    template<typename T>
    T* getActiveSensor()
    {
        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            SENSSimulatedBase* base    = _activeSensors[i].get();
            T*                 derived = dynamic_cast<T*>(base);
            if (derived)
                return derived;
        }

        return nullptr;
    }

    SENSSimulatedGps*         getGpsSensorPtr() { return getActiveSensor<SENSSimulatedGps>(); }
    SENSSimulatedOrientation* getOrientationSensorPtr() { return getActiveSensor<SENSSimulatedOrientation>(); }
    SENSSimulatedCamera*      getCameraSensorPtr() { return getActiveSensor<SENSSimulatedCamera>(); }

    bool isRunning() const { return _running; }

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
        bool aSensorIsRunning = false;
        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            if (_activeSensors[i]->isThreadRunning())
            {
                aSensorIsRunning = true;
                break;
            }
        }

        if (!aSensorIsRunning)
            _running = false;
    }

    void estimateSimStartPoint()
    {
        //search for earliest time point
        bool initialized = false;

        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            const SENSTimePt& tp = _activeSensors[i]->firstTimePt();
            if (!initialized)
            {
                initialized     = true;
                _simStartTimePt = tp;
            }
            else if (tp < _simStartTimePt)
                _simStartTimePt = tp;
        }

        for (int i = 0; i < _activeSensors.size(); ++i)
            _activeSensors[i]->setCommonSimStartTimePt(_simStartTimePt);
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
                        readTimePt       = std::stol(values[0]);
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
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedGps>(
                      new SENSSimulatedGps(std::bind(&SENSSimulator::onStart, this),
                                           std::bind(&SENSSimulator::onSensorSimStopped, this),
                                           std::move(data))));
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
                        readTimePt = std::stol(values[0]);
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
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedOrientation>(
                      new SENSSimulatedOrientation(std::bind(&SENSSimulator::onStart, this),
                                                   std::bind(&SENSSimulator::onSensorSimStopped, this),
                                                   std::move(data))));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s", orientationFileName.c_str());
        }
    }

    void loadCameraData(const std::string& dirName)
    {
        std::string textFileName  = dirName + "camera.txt";
        std::string videoFileName = dirName + "video.avi";

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
                    if (values.size() == 2)
                    {
                        readTimePt = std::stol(values[0]);
                        frameIndex = std::stoi(values[1]);

                        SENSMicroseconds readTimePtUs(readTimePt);
                        SENSTimePt       tPt(readTimePtUs);
                        data.push_back(std::make_pair(tPt, frameIndex));
                    }
                }
                file.close();

                if (data.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedCamera>(
                      new SENSSimulatedCamera(std::bind(&SENSSimulator::onStart, this),
                                              std::bind(&SENSSimulator::onSensorSimStopped, this),
                                              std::move(data),
                                              videoFileName)));
                }
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s or ", textFileName.c_str(), videoFileName.c_str());
        }
    }

    std::vector<std::unique_ptr<SENSSimulatedBase>> _activeSensors;

    //real start time point (when SENSSimulation::start was called)
    SENSTimePt _startTimePoint;
    //start time point of simulation
    SENSTimePt _simStartTimePt;

    std::atomic_bool _running{false};
};

#endif
