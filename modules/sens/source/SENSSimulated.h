#ifndef SENS_SIMULATED_H
#define SENS_SIMULATED_H

#include <functional>
#include <memory>
#include <chrono>
#include <condition_variable>

#include <Utils.h>
#include <SENSSimClock.h>
#include <SENSCamera.h>
#include <SENSGps.h>
#include <SENSOrientation.h>

class SENSSimulator;

//-----------------------------------------------------------------------------
/*! SENSSimulatedBase
This ia pure virtual base class to control SENSSimulated implementations via a common interface in SENSSimulator
 */
class SENSSimulatedBase
{
public:
    virtual ~SENSSimulatedBase() {}
    //!indicates if simulation thread is running
    virtual bool isThreadRunning() const = 0;

    virtual bool getErrorMsg(std::string& msg) = 0;
};

//-----------------------------------------------------------------------------
/*! SENSSimulated
Implements the sensor simulation backend. SENSSimulated runs a thread (see feedSensor()) that calls feedSensorData() interface
function which feeds the next data value to a sensor (e.g. SENSGps). The current simulation time is retrieved from
the SENSSimulator clock and depending on this the next data value is selected and fed a the respective time.
This class contains common functionality for SENSSimulated implementations.
 */
template<typename T>
class SENSSimulated : public SENSSimulatedBase
{
public:
    virtual ~SENSSimulated() {}

protected:
    using StartSimCB         = std::function<void(void)>;
    using SensorSimStoppedCB = std::function<void(void)>;

    //!Transfer callbacks to SENSSimulator during construction. The SENSSimulator is informed by
    //!these if the SENSSimulated state changes (start, stop). We use callbacks to avoid circular dependencies.
    SENSSimulated(const std::string                       name,
                  StartSimCB                              startSimCB,
                  SensorSimStoppedCB                      sensorSimStoppedCB,
                  std::vector<std::pair<SENSTimePt, T>>&& data,
                  const SENSSimClock&                     clock);

    //!start the sensor simulation thread
    void startSim();
    //!stop the sensor simulation thread
    void stopSim();

    //!get error msg (valid if function returns true)
    bool getErrorMsg(std::string& msg) override;

    bool isThreadRunning() const override { return _threadIsRunning; }

    //!feed new sensor data to sensor
    virtual void feedSensorData(const int counter) = 0;
    //!prepare things that may take some time for the next writing of sensor data
    //!(e.g. for video reading we can already read the frame and do decoding and maybe it also blocks for some reasons..)
    virtual void prepareSensorData(const int counter){};
    //!special treatment when there is latency (e.g. when feeding camera frames from cv::videocapture updating the frame pos may take ages)
    virtual void onLatencyProblem(int& counter) {}

private:
    //!thread run routine to feed sensor with data.
    void feedSensor();

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

    const SENSSimClock& _clock;

    std::string _name;

    std::mutex  _msgMutex;
    std::string _errorMsg;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedGps
SENSSimulated implementation for GPS sensor simulation. Implements the SENSGps interface
 to make it a full gps sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedGps : public SENSGps
  , public SENSSimulated<SENSGps::Location>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedGps();

    bool start() override;
    void stop() override;

private:
    //only SENSSimulator can instantiate
    SENSSimulatedGps(StartSimCB                                              startSimCB,
                     SensorSimStoppedCB                                      sensorSimStoppedCB,
                     std::vector<std::pair<SENSTimePt, SENSGps::Location>>&& data,
                     const SENSSimClock&                                     clock);

    void feedSensorData(const int counter) override;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedOrientation
SENSSimulated implementation for Orientation sensor simulation. Implements the SENSOrientation interface
 to make it a full orientation sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedOrientation : public SENSOrientation
  , public SENSSimulated<SENSOrientation::Quat>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedOrientation();

    bool start() override;
    void stop() override;

private:
    SENSSimulatedOrientation(StartSimCB                                                  startSimCB,
                             SensorSimStoppedCB                                          sensorSimStoppedCB,
                             std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>&& data,
                             const SENSSimClock&                                         clock);

    void feedSensorData(const int counter) override;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedCamera
SENSSimulated implementation for Camera sensor simulation. Implements the SENSBaseCamera interface
to make it a full camera sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedCamera : public SENSBaseCamera
  , public SENSSimulated<int>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedCamera();

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics = true) override;

    void stop() override;

    const SENSCaptureProps& captureProperties() override;

private:
    SENSSimulatedCamera(StartSimCB                                startSimCB,
                        SensorSimStoppedCB                        sensorSimStoppedCB,
                        std::vector<std::pair<SENSTimePt, int>>&& data,
                        std::string                               videoFileName,
                        SENSCameraConfig                          cameraConfig,
                        const SENSSimClock&                       clock);

    void feedSensorData(const int counter) override;
    void prepareSensorData(const int counter) override;
    //!when feeding camera frames from cv::videocapture updating the frame pos may take ages. So we skip one more frame and update the next frame counter
    void onLatencyProblem(int& counter) override;

    std::string      _videoFileName;
    cv::VideoCapture _cap;

    cv::Mat    _preparedFrame;
    int        _preparedFrameIndex = -1;
    cv::Mat    _frame;
    std::mutex _frameMutex;
};

#endif
