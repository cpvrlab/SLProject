#include "SENSSimulated.h"

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
    //start the simulation thread
    _thread = std::thread(&SENSSimulated::feedSensor, this /*, startTimePt, _commonSimStartTimePt*/);
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
void SENSSimulated<T>::feedSensor(/*const SENSTimePt startTimePt, const SENSTimePt simStartTimePt*/)
{
    _threadIsRunning = true;
    int counter      = 0;

    //auto       passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
    //SENSTimePt simTime       = simStartTimePt + passedSimTime;
    SENSTimePt simTime = _clock.now();

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
        SENSTimePt simTime = _clock.now();

        //process late values
        int i = 0;
        while (counter < _data.size() && _data[counter].first < simTime)
        {
            //SENS_DEBUG("feed sensor index %d with latency: %d us, SimT: %d, ValueT: %d", counter, std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count(), std::chrono::time_point_cast<SENSMicroseconds>(simTime).time_since_epoch().count(), std::chrono::time_point_cast<SENSMicroseconds>(_data[counter].first).time_since_epoch().count());
            SENS_DEBUG("feed %s sensor index %d with latency: %d us", _name.c_str(), counter, std::chrono::duration_cast<SENSMicroseconds>(_data[counter].first - simTime).count());
            feedSensorData(counter);

            //setting the location maybe took some time, so we update simulation time
            simTime = _clock.now();
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

        //passedSimTime = std::chrono::duration_cast<SENSMicroseconds>((SENSTimePt)SENSClock::now() - startTimePt);
        //simTime       = simStartTimePt + passedSimTime;
        simTime = _clock.now();

        //locationsCounter should now point to a value in the simulation future so lets wait
        const SENSTimePt& valueTime = _data[counter].first;

        //simTime should now be smaller than valueTime because valueTime is in the simulation future
        SENSMicroseconds waitTimeUs = std::chrono::duration_cast<SENSMicroseconds>(valueTime - simTime);
        //We reduce the wait time because thread sleep is not very exact (best is not to wait at all)
        SENSMicroseconds reducedWaitTimeUs((long)(0.001 * (double)waitTimeUs.count()));

        //HighResTimer t;
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
                                                   cv::Size                      imgBGRSize,
                                                   bool                          mirrorV,
                                                   bool                          mirrorH,
                                                   bool                          convToGrayToImgManip,
                                                   int                           imgManipWidth,
                                                   bool                          provideIntrinsics,
                                                   float                         fovDegFallbackGuess)
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

    cv::Size imgManipSize;
    if (imgManipWidth > 0)
        imgManipSize = {imgManipWidth, (int)((float)imgManipWidth * (float)targetSize.height / (float)targetSize.width)};
    else
        imgManipSize = targetSize;

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

void SENSSimulatedCamera::stop()
{
    if (_started)
    {
        stopSim();
        _started = false;
    }
}

SENSFramePtr SENSSimulatedCamera::latestFrame()
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

const SENSCaptureProperties& SENSSimulatedCamera::captureProperties()
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

SENSSimulatedCamera::SENSSimulatedCamera(StartSimCB                                startSimCB,
                                         SensorSimStoppedCB                        sensorSimStoppedCB,
                                         std::vector<std::pair<SENSTimePt, int>>&& data,
                                         std::string                               videoFileName,
                                         const SENSSimClock&                       clock)
  : SENSSimulated("camera", startSimCB, sensorSimStoppedCB, std::move(data), clock),
    _videoFileName(videoFileName)
{
    _permissionGranted = true;
}

void SENSSimulatedCamera::feedSensorData(const int counter)
{
    //prepare if not yet prepared (e.g. when while loop writes multiple lines)
    if (_data[counter].second != _preparedFrameIndex)
    {
        prepareSensorData(counter);
    }

    //todo: move to base class
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        if (_listeners.size())
        {
            SENSTimePt timePt = SENSClock::now();
            for (SENSCameraListener* l : _listeners)
                l->onFrame(timePt, _preparedFrame.clone());
        }
    }

    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        _frame = _preparedFrame;
    }
}

void SENSSimulatedCamera::prepareSensorData(const int counter)
{
    int frameIndex = _data[counter].second;
    if (frameIndex == _preparedFrameIndex)
        return;

    _cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);

    cv::Mat frame;
    //HighResTimer t;
    _cap.read(frame);
    //SENS_DEBUG("read time index %d: %d", frameIndex, t.elapsedTimeInMicroSec());
    {
        _preparedFrameIndex = frameIndex;
        std::lock_guard<std::mutex> lock(_frameMutex);
        _preparedFrame = frame;
    }
}
