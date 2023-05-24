#include "SENSSimulated.h"
#include "HighResTimer.h"

//-----------------------------------------------------------------------------
template<typename T>
SENSSimulated<T>::SENSSimulated(const std::string                       name,
                                StartSimCB                              startSimCB,
                                SensorSimStoppedCB                      sensorSimStoppedCB,
                                std::vector<std::pair<SENSTimePt, T>>&& data,
                                const SENSSimClock&                     clock)
  : _name(name),
    _startSimCB(startSimCB),
    _sensorSimStoppedCB(sensorSimStoppedCB),
    _data(data),
    _clock(clock)
{
}

template<typename T>
void SENSSimulated<T>::startSim()
{
    //inform SENSSimulator about start
    _startSimCB();
    //stop the local simulation thread if running
    stopSim();
    _errorMsg.clear();
    //start the simulation thread
    _thread = std::thread(&SENSSimulated::feedSensor, this);
}

template<typename T>
void SENSSimulated<T>::stopSim()
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

template<typename T>
bool SENSSimulated<T>::getErrorMsg(std::string& msg)
{
    std::lock_guard<std::mutex> lock(_msgMutex);
    if (_errorMsg.empty())
        return false;
    else
    {
        msg = _errorMsg;
        return true;
    }
}

template<typename T>
void SENSSimulated<T>::feedSensor()
{
    _threadIsRunning = true;
    //index of current data set to feed
    int counter = 0;

    SENSTimePt simTime = _clock.now();
    //bring counter close to startTimePt (maybe we skip some values to synchronize simulated sensors)
    while (counter < _data.size() && _data[counter].first < simTime)
        counter++;

    if (counter >= _data.size())
    {
        //end of simulation
        _threadIsRunning = false;
        _sensorSimStoppedCB();
        return;
    }

    while (true)
    {
        //estimate current sim time
        SENSTimePt simTime = _clock.now();

        //feed latest data
        int oldCounter = counter;
        if (counter < _data.size() && _data[counter].first <= simTime)
        {
            //SENS_DEBUG("feed sensor index %d with latency: %d us, SimT: %d, ValueT: %d", counter, std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count(), std::chrono::time_point_cast<SENSMicroseconds>(simTime).time_since_epoch().count(), std::chrono::time_point_cast<SENSMicroseconds>(_data[counter].first).time_since_epoch().count());
            SENS_DEBUG("feed %s sensor index %d with latency: %d us", _name.c_str(), counter, std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count());
            feedSensorData(counter);
            if (_stop)
                break;

            //update the counter
            simTime = _clock.now();
            while (counter < _data.size() && _data[counter].first < simTime)
                counter++;
        }

        if (counter - oldCounter > 1)
        {
            //for feeding camera frames from cv::videocapture we need a special treatment in this case
            //(THIS FUNCTION MAY CHANGE THE COUNTER)
            onLatencyProblem(counter);
            //report an error
            int numberOfSkippedData = counter - oldCounter;

            std::stringstream ss;
            ss << "Data feeding is too slow. Skipped " << numberOfSkippedData << " data sets!";
            SENS_WARN("SENSSimulated feedSensor: %s", ss.str().c_str());
            {
                std::lock_guard<std::mutex> lock(_msgMutex);
                _errorMsg = ss.str();
            }
        }

        //end of simulation
        if (counter >= _data.size())
            break;

        //prepare things that may take some time (e.g. for video reading we can already read the frame and do decoding and maybe it also blocks for some reasons..)
        prepareSensorData(counter);

        //WAITING DOES NOT WORK VERY EXACTLY ON WINDOWS (it does wake too late)
#ifdef WAIT
        simTime = _clock.now();
        //simTime should now be smaller than valueTime because valueTime is in the simulation future
        SENSMicroseconds waitTimeUs = std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime);
        //We reduce the wait time because thread sleep is not very exact (best is not to wait at all)
        SENSMicroseconds reducedWaitTimeUs((long)(0.01 * (double)waitTimeUs.count()));

        //HighResTimer t;
        std::unique_lock<std::mutex> lock(_mutex);
        _condVar.wait_for(lock, reducedWaitTimeUs, [&]
                          { return _stop == true; });
        //SENS_DEBUG("wait time %d us", waitTimeUs.count());
        //SENS_DEBUG("woke after %d us", t.elapsedTimeInMicroSec());
#endif
        if (_stop)
            break;
    }

    _threadIsRunning = false;
    _sensorSimStoppedCB();
}

//explicit template instantiation
template class SENSSimulated<SENSGps::Location>;
template class SENSSimulated<SENSOrientation::Quat>;
template class SENSSimulated<int>;

//-----------------------------------------------------------------------------

SENSSimulatedGps::~SENSSimulatedGps()
{
    stop();
}

bool SENSSimulatedGps::start()
{
    if (!_running)
    {
        startSim();
        _running = true;
    }

    return _running;
}

void SENSSimulatedGps::stop()
{
    if (_running)
    {
        stopSim();
        _running = false;
    }
}

//only SENSSimulator can instantiate
SENSSimulatedGps::SENSSimulatedGps(StartSimCB                                              startSimCB,
                                   SensorSimStoppedCB                                      sensorSimStoppedCB,
                                   std::vector<std::pair<SENSTimePt, SENSGps::Location>>&& data,
                                   const SENSSimClock&                                     clock)
  : SENSSimulated("gps", startSimCB, sensorSimStoppedCB, std::move(data), clock)
{
    _permissionGranted = true;
}

void SENSSimulatedGps::feedSensorData(const int counter)
{
    setLocation(_data[counter].second);
}

//-----------------------------------------------------------------------------
SENSSimulatedOrientation::~SENSSimulatedOrientation()
{
    stop();
}

bool SENSSimulatedOrientation::start()
{
    if (!_running)
    {
        startSim();
        _running = true;
    }

    return _running;
}

void SENSSimulatedOrientation::stop()
{
    if (_running)
    {
        stopSim();
        _running = false;
    }
}

SENSSimulatedOrientation::SENSSimulatedOrientation(StartSimCB                                                  startSimCB,
                                                   SensorSimStoppedCB                                          sensorSimStoppedCB,
                                                   std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>&& data,
                                                   const SENSSimClock&                                         clock)
  : SENSSimulated("orientation", startSimCB, sensorSimStoppedCB, std::move(data), clock)
{
}

void SENSSimulatedOrientation::feedSensorData(const int counter)
{
    setOrientation(_data[counter].second);
}

//-----------------------------------------------------------------------------
SENSSimulatedCamera::~SENSSimulatedCamera()
{
    stop();
}

const SENSCameraConfig& SENSSimulatedCamera::start(std::string                   deviceId,
                                                   const SENSCameraStreamConfig& streamConfig,
                                                   bool                          provideIntrinsics)
{
    if (_started)
    {
        Utils::warnMsg("SENSWebCamera", "Call to start was ignored. Camera is currently running!", __LINE__, __FILE__);
        return _config;
    }

    if (!_cap.isOpened())
    {
        if (!_cap.open(_videoFileName))
            throw SENSException(SENSType::CAM, "Could not open camera simulator for filename: " + _videoFileName, __LINE__, __FILE__);
    }

    //retrieve all camera characteristics
    if (_captureProperties.size() == 0)
        captureProperties();

    //CONFIG SHOULD HAVE BEEN LOADED FROM FILE

    processStart();

    //start the sensor simulation
    startSim();

    _started = true;

    return _config;
}

void SENSSimulatedCamera::stop()
{
    if (_started)
    {
        stopSim();
        _started = false;
    }
}

const SENSCaptureProps& SENSSimulatedCamera::captureProperties()
{
    if (!_captureProperties.size())
    {
        SENSCameraDeviceProps characteristics(_config.deviceId, _config.facing);
        characteristics.add(_config.streamConfig.widthPix, _config.streamConfig.heightPix, _config.streamConfig.focalLengthPix);
        _captureProperties.push_back(characteristics);
    }

    return _captureProperties;
}

SENSSimulatedCamera::SENSSimulatedCamera(StartSimCB                                startSimCB,
                                         SensorSimStoppedCB                        sensorSimStoppedCB,
                                         std::vector<std::pair<SENSTimePt, int>>&& data,
                                         std::string                               videoFileName,
                                         SENSCameraConfig                          cameraConfig,
                                         const SENSSimClock&                       clock)
  : SENSSimulated("camera", startSimCB, sensorSimStoppedCB, std::move(data), clock),
    _videoFileName(videoFileName)
{
    _config            = cameraConfig;
    _permissionGranted = true;
}

void SENSSimulatedCamera::feedSensorData(const int counter)
{
    HighResTimer t;
    //prepare if not yet prepared (e.g. when while loop writes multiple lines)
    if (_data[counter].second != _preparedFrameIndex)
    {
        SENS_DEBUG("preparing in feedSensorData");
        prepareSensorData(counter);
    }

    updateFrame(_preparedFrame, cv::Mat(), false, _preparedFrame.size().width, _preparedFrame.size().height);

    SENS_DEBUG("feedSensorData %lld us", t.elapsedTimeInMicroSec());
}

void SENSSimulatedCamera::prepareSensorData(const int counter)
{
    if (counter >= _data.size())
        return;

    int frameIndex = _data[counter].second;
    if (frameIndex == _preparedFrameIndex)
        return;

    HighResTimer t;

    int nextFramePos = _cap.get(cv::CAP_PROP_POS_FRAMES);
    if (nextFramePos != frameIndex)
    {
        SENS_DEBUG("updating frame pos");
        _cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    }

    cv::Mat frame;

    _cap.read(frame);

    {
        _preparedFrameIndex = frameIndex;
        std::lock_guard<std::mutex> lock(_frameMutex);
        _preparedFrame = frame;
    }

    SENS_DEBUG("prepared index %d: %lld", frameIndex, t.elapsedTimeInMicroSec());
}

void SENSSimulatedCamera::onLatencyProblem(int& counter)
{
    //Setting the next frame position may take a lot of time.
    //Thats why we need this special treatment for camera feed.

    //update the frame pos with current counter and measure needed time
    SENSTimePt t0 = SENSClock::now();
    prepareSensorData(counter);
    SENSTimePt       t1         = SENSClock::now();
    SENSMicroseconds neededTime = std::chrono::duration_cast<SENSMicroseconds>(t1 - t0);

    //SENS_DEBUG("update CAP_PROP_POS_FRAMES %d: %lld", counter, t.elapsedTimeInMicroSec());
    counter++;
    SENSTimePt now = _clock.now();
    while (counter < _data.size() - 1 && _data[counter].first < now)
    {
        //when we enter this loop the frame the neededTime for pos update took longer than the availableTime between too values.
        //we measure how much longer and update the counter accordingly
        SENSMicroseconds availableTime = std::chrono::duration_cast<SENSMicroseconds>(_data[counter + 1].first - _data[counter].first);
        auto             factor        = neededTime.count() / availableTime.count();
        factor++;
        SENS_DEBUG("factor %lld neededTime %lld available time %lld", factor, neededTime.count(), availableTime.count());
        counter += factor;

        if (counter < _data.size())
        {
            t0 = SENSClock::now();
            prepareSensorData(counter);
            t1         = SENSClock::now();
            neededTime = std::chrono::duration_cast<SENSMicroseconds>(t1 - t0);
        }

        now = _clock.now();
    }
}
