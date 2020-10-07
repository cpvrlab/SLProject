#ifndef SENS_RECORDER_DATAHANDLER_H
#define SENS_RECORDER_DATAHANDLER_H

#include <string>
#include <thread>
#include <atomic>
#include <deque>
#include <utility>
#include <mutex>
#include <fstream>
//for video writer on android
#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <Utils.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <sens/SENSCamera.h>

using GpsInfo         = std::pair<SENSGps::Location, SENSTimePt>;
using OrientationInfo = std::pair<SENSOrientation::Quat, SENSTimePt>;
using FrameInfo       = std::pair<cv::Mat, SENSTimePt>;

//-----------------------------------------------------------------------------
/*! SENSRecorderDataHandler
 This class is meant to be used exclusively by the SENSRecorder class. The SENSRecorder listens to sensors
 and informs SENSRecorderDataHandler backends about new data. The SENSRecorderDataHandler stores values to file.
 */
template<typename T>
class SENSRecorderDataHandler
{
public:
    SENSRecorderDataHandler(const std::string& name);
    virtual ~SENSRecorderDataHandler();

    //!start the store thread
    void start(const std::string& outputDir);
    //!stop the store thread and clear values in queue
    void stop();
    //!add new value to store queue
    void add(T&& item);

protected:
    //!called in thread store routine when thread starts
    virtual void writeOnThreadStart() {}
    //!called in thread store routine when file could be opened
    virtual void writeHeaderToFile(ofstream& file) {}
    //!called in thread store routine for every new line
    virtual void writeLineToFile(ofstream& file, const T& data) = 0;
    //!called in thread store routine when thread finished
    virtual void writeOnThreadFinish() {}

    //!output directory
    std::string _outputDir;

private:
    void store();

    //new data queue: values are added by SENSRecorder and retrieved and
    //written by store thread
    std::deque<T> _queue;
    //condition variable and mutex for store thread
    std::mutex              _mutex;
    std::condition_variable _condVar;
    //store thread
    std::thread _thread;
    //stop store thread
    bool _stop = false;
    //name of this handler (e.g. gps)
    std::string _name;
};

//-----------------------------------------------------------------------------
/*! SENSGpsRecorderDataHandler
 */
class SENSGpsRecorderDataHandler : public SENSRecorderDataHandler<GpsInfo>
{
public:
    SENSGpsRecorderDataHandler();
    void writeLineToFile(ofstream& file, const GpsInfo& data) override;
};

//-----------------------------------------------------------------------------
/*! SENSOrientationRecorderDataHandler
 */
class SENSOrientationRecorderDataHandler : public SENSRecorderDataHandler<OrientationInfo>
{
public:
    SENSOrientationRecorderDataHandler();
    void writeLineToFile(ofstream& file, const OrientationInfo& data) override;
};

//-----------------------------------------------------------------------------
/*! SENSCameraRecorderDataHandler
 */
class SENSCameraRecorderDataHandler : public SENSRecorderDataHandler<FrameInfo>
{
public:
    SENSCameraRecorderDataHandler();

    void writeOnThreadStart() override;
    void writeLineToFile(ofstream& file, const FrameInfo& data) override;
    void writeOnThreadFinish() override;

    void setCalibration(const SENSCalibration& calibration);

private:
    cv::VideoWriter _videoWriter;
    int             _frameIndex = 0;

    std::unique_ptr<SENSCalibration> _calibration;
    std::mutex                       _calibrationMutex;
    bool                             _calibrationChanged = false;
};

#endif
