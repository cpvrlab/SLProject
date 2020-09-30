#ifndef SENS_RECORDER_H
#define SENS_RECORDER_H

#include <string>
#include <thread>
#include <atomic>
#include <deque>
#include <utility>
#include <mutex>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <sens/SENSCamera.h>
#include <Utils.h>

using GpsInfo         = std::pair<SENSGps::Location, SENSTimePt>;
using OrientationInfo = std::pair<SENSOrientation::Quat, SENSTimePt>;
using FrameInfo       = std::pair<cv::Mat, SENSTimePt>;

template<typename T>
class SENSRecorderDataHandler
{
public:
    SENSRecorderDataHandler(const std::string& name)
      : _name(name)
    {
    }

    virtual ~SENSRecorderDataHandler()
    {
        stop();
    }

    void stop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _stop = true;
        lock.unlock();
        _condVar.notify_one();

        if (_thread.joinable())
            _thread.join();

        lock.lock();
        _queue.clear();
        _stop = false;
    }

    void start(const std::string& outputDir)
    {
        stop();
        //start writer thread
        _outputDir = outputDir;
        _thread    = std::thread(&SENSRecorderDataHandler::store, this);
    }

    void store()
    {
        writeOnThreadStart();
        //open file
        std::string fileName = _outputDir + _name + ".txt";
        ofstream    file;
        file.open(fileName);
        if (file.is_open())
        {
            writeHeaderToFile(file);
            while (true)
            {
                std::unique_lock<std::mutex> lock(_mutex);

                _condVar.wait(lock, [&] { return (_stop == true || _queue.size() != 0); });
                if (_stop)
                    break;

                std::deque<T> localQueue;
                while (_queue.size())
                {
                    localQueue.push_back(_queue.front());
                    _queue.pop_front();
                }

                lock.unlock();

                //write data
                while (localQueue.size())
                {
                    writeLineToFile(file, localQueue.front());
                    localQueue.pop_front();
                }
            }
        }
        writeOnThreadFinish();
        //close file
        file.close();
    }

    void add(T&& item)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.push_back(std::move(item));
        lock.unlock();
        _condVar.notify_one();
    }

    virtual void writeOnThreadStart() {}
    virtual void writeHeaderToFile(ofstream& file) {}
    virtual void writeLineToFile(ofstream& file, const T& data) = 0;
    virtual void writeOnThreadFinish() {}

protected:
    std::string _outputDir;

private:
    std::deque<T>           _queue;
    std::mutex              _mutex;
    std::condition_variable _condVar;
    std::thread             _thread;
    bool                    _stop = false;

    std::string _name;
    //std::function<void(T&, ofstream&)> _writeDataToFile;
};

class SENSGpsRecorderDataHandler : public SENSRecorderDataHandler<GpsInfo>
{
public:
    SENSGpsRecorderDataHandler()
      : SENSRecorderDataHandler("gps")
    {
    }

    void writeLineToFile(ofstream& file, const GpsInfo& data) override
    {
        file << std::chrono::time_point_cast<SENSMicroseconds>(data.second).time_since_epoch().count() << " "
             << data.first.latitudeDEG << " "
             << data.first.longitudeDEG << " "
             << data.first.altitudeM << " "
             << data.first.accuracyM << "\n";
    }

private:
};

class SENSOrientationRecorderDataHandler : public SENSRecorderDataHandler<OrientationInfo>
{
public:
    SENSOrientationRecorderDataHandler()
      : SENSRecorderDataHandler("orientation")
    {
    }

    void writeLineToFile(ofstream& file, const OrientationInfo& data) override
    {
        //reading (https://stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back)
        //long readTimePt;
        //microseconds readTimePtUs(readTimePt);
        //time_point<high_resolution_clock> dt(readTimePt);

        file << std::chrono::time_point_cast<SENSMicroseconds>(data.second).time_since_epoch().count() << " "
             << data.first.quatX << " "
             << data.first.quatY << " "
             << data.first.quatZ << " "
             << data.first.quatW << "\n";
    }

private:
};

class SENSCameraRecorderDataHandler : public SENSRecorderDataHandler<FrameInfo>
{
public:
    SENSCameraRecorderDataHandler()
      : SENSRecorderDataHandler("camera")
    {
    }

    void writeOnThreadStart() override
    {
        std::lock_guard<std::mutex> lock(_calibrationMutex);
        if (_calibration)
        {
            _calibration->save(_outputDir, "calibration.json");
            _calibrationChanged = false;
        }
    }

    void writeLineToFile(ofstream& file, const FrameInfo& data) override
    {
        //write time and frame index
        file << std::chrono::time_point_cast<SENSMicroseconds>(data.second).time_since_epoch().count() << " "
             << _frameIndex << "\n";
        _frameIndex++;

        if (!_videoWriter.isOpened())
            _videoWriter.open(_outputDir + "video.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, data.first.size(), true);

        //store frame to video capture
        _videoWriter.write(data.first);

        if (_calibrationChanged)
        {
            std::lock_guard<std::mutex> lock(_calibrationMutex);
            _calibration->save(_outputDir, "calibration.json");
            _calibrationChanged = false;
        }
    }
    
    void writeOnThreadFinish() override
    {
        if(_videoWriter.isOpened())
            _videoWriter.release();
    }

    void setCalibration(const SENSCalibration& calibration)
    {
        std::lock_guard<std::mutex> lock(_calibrationMutex);
        _calibration        = std::make_unique<SENSCalibration>(calibration.cameraMat(),
                                                         calibration.distortion(),
                                                         calibration.imageSize(),
                                                         calibration.boardSize(),
                                                         calibration.boardSquareMM(),
                                                         calibration.reprojectionError(),
                                                         calibration.numCapturedImgs(),
                                                         calibration.calibrationTime(),
                                                         -1,
                                                         calibration.isMirroredH(),
                                                         calibration.isMirroredV(),
                                                         calibration.camType(),
                                                         calibration.computerInfos(),
                                                         calibration.calibrationFlags(),
                                                         calibration.undistortMapsValid());
        _calibrationChanged = true;
    }

private:
    cv::VideoWriter _videoWriter;
    int             _frameIndex = 0;

    std::unique_ptr<SENSCalibration> _calibration;
    std::mutex                       _calibrationMutex;
    bool                             _calibrationChanged = false;
};

class SENSRecorder : public SENSGpsListener
  , public SENSOrientationListener
  , public SENSCameraListener
{
    using GpsInfo         = std::pair<SENSGps::Location, SENSTimePt>;
    using OrientationInfo = std::pair<SENSOrientation::Quat, SENSTimePt>;
    using FrameInfo       = std::pair<SENSFramePtr, SENSTimePt>;

public:
    SENSRecorder(const std::string& outputDir)
      : _outputDir(outputDir)
    {
        if (Utils::dirExists(outputDir))
        {
            _gpsDataHandler         = new SENSGpsRecorderDataHandler();
            _orientationDataHandler = new SENSOrientationRecorderDataHandler();
            _cameraDataHandler      = new SENSCameraRecorderDataHandler();
        }
        else
            Utils::log("SENS", "SENSRecorder: Directory does not exist: %s", outputDir.c_str());
    }
    ~SENSRecorder()
    {
        stop();

        if (_gpsDataHandler)
            delete _gpsDataHandler;
        if (_orientationDataHandler)
            delete _orientationDataHandler;
        if (_cameraDataHandler)
            delete _cameraDataHandler;
    }

    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    bool activateGps(SENSGps* sensor)
    {
        if (!_running && sensor)
        {
            deactivateGps();
            _gps = sensor;
            return true;
        }
        else
            return false;
    }
    bool deactivateGps()
    {
        if (!_running && _gps)
        {
            _gps = nullptr;
            return true;
        }
        else
            return false;
    }
    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    bool activateOrientation(SENSOrientation* sensor)
    {
        if (!_running && sensor)
        {
            deactivateOrientation();
            _orientation = sensor;
            return true;
        }
        else
            return false;
    }

    bool deactivateOrientation()
    {
        if (!_running && _orientation)
        {
            _orientation = nullptr;
            return true;
        }
        else
            return true;
    }

    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    bool activateCamera(SENSCamera* sensor)
    {
        if (!_running && sensor)
        {
            deactivateCamera();
            _camera = sensor;
            return true;
        }
        else
            return false;
    }

    bool deactivateCamera()
    {
        if (!_running && _camera)
        {
            _camera = nullptr;
            return true;
        }
        else
            return true;
    }

    bool start()
    {
        if (_running)
            return false;

        std::string recordDir = Utils::unifySlashes(_outputDir) + Utils::getDateTime2String() + "_SENSRecorder/";
        if (!Utils::makeDir(recordDir))
        {
            Utils::log("SENS", "SENSRecorder start: could not create record directory: %s", recordDir.c_str());
            return false;
        }

        if (_gps && _gpsDataHandler)
        {
            _gps->registerListener(this);
            _gpsDataHandler->start(recordDir);
            _running = true;
        }

        if (_orientation && _orientationDataHandler)
        {
            _orientation->registerListener(this);
            _orientationDataHandler->start(recordDir);
            _running = true;
        }

        if (_camera && _cameraDataHandler)
        {
            _camera->registerListener(this);
            _cameraDataHandler->start(recordDir);
            _running = true;
        }

        return _running;
    }

    void stop()
    {
        if (_gps)
        {
            _gps->unregisterListener(this);
            if (_gpsDataHandler)
                _gpsDataHandler->stop();
        }

        if (_orientation)
        {
            _orientation->unregisterListener(this);
            if (_orientationDataHandler)
                _orientationDataHandler->stop();
        }

        if (_camera)
        {
            _camera->unregisterListener(this);
            if (_cameraDataHandler)
                _cameraDataHandler->stop();
        }

        _running = false;
    }

    bool isRunning() { return _running; }

private:
    void onGps(const SENSTimePt& timePt, const SENSGps::Location& loc) override
    {
        auto newData = std::make_pair(loc, timePt);
        if (_gpsDataHandler)
            _gpsDataHandler->add(std::move(newData));
    }

    void onOrientation(const SENSTimePt& timePt, const SENSOrientation::Quat& ori) override
    {
        auto newData = std::make_pair(ori, timePt);
        if (_orientationDataHandler)
            _orientationDataHandler->add(std::move(newData));
    }

    void onFrame(const SENSTimePt& timePt, cv::Mat frame) override
    {
        auto newData = std::make_pair(frame, timePt);
        if (_cameraDataHandler)
            _cameraDataHandler->add(std::move(newData));
    }

    void onCalibrationChanged(const SENSCalibration& calibration) override
    {
        _cameraDataHandler->setCalibration(calibration);
    }

    std::string _outputDir;

    SENSGpsRecorderDataHandler*         _gpsDataHandler         = nullptr;
    SENSOrientationRecorderDataHandler* _orientationDataHandler = nullptr;
    SENSCameraRecorderDataHandler*      _cameraDataHandler      = nullptr;

    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;
    SENSCamera*      _camera      = nullptr;

    std::atomic_bool _running{false};
};

#endif
