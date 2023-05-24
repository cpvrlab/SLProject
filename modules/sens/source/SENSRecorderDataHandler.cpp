#include "SENSRecorderDataHandler.h"
#include <HighResTimer.h>
//-----------------------------------------------------------------------------
template<typename T>
SENSRecorderDataHandler<T>::SENSRecorderDataHandler(const std::string& name)
  : _name(name)
{
}

template<typename T>
SENSRecorderDataHandler<T>::~SENSRecorderDataHandler()
{
    stop();
}

template<typename T>
void SENSRecorderDataHandler<T>::start(const std::string& outputDir)
{
    stop();
    //start writer thread
    _errorMsg.clear();
    _outputDir = outputDir;
    _thread    = std::thread(&SENSRecorderDataHandler::store, this);
}

template<typename T>
void SENSRecorderDataHandler<T>::stop()
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

template<typename T>
void SENSRecorderDataHandler<T>::store()
{
    SENS_DEBUG("SENSRecorderDataHandler: starting store");
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

            _condVar.wait(lock, [&]
                          { return (_stop == true || _queue.size() != 0); });
            if (_stop)
                break;

            SENS_DEBUG("SENSRecorderDataHandler: queue size: %d", _queue.size());

            std::deque<T> localQueue;
            while (_queue.size())
            {
                localQueue.push_back(_queue.front());
                _queue.pop_front();
            }

            lock.unlock();

            //write data
            if (localQueue.size())
            {
                writeLineToFile(file, localQueue.front());
                localQueue.pop_front();
            }

            if (localQueue.size())
            {
                std::stringstream ss;
                ss << "Data writing is too slow. Skipping " << localQueue.size() << " values from queue!";
                SENS_WARN("SENSRecorderDataHandler store: %s", ss.str().c_str());
                //if buffer is running full, warn and skip values
                {
                    std::lock_guard<std::mutex> lock(_msgMutex);
                    _errorMsg = ss.str();
                }
                localQueue.clear();
            }
        }
    }
    writeOnThreadFinish();
    //close file
    file.close();
}

template<typename T>
void SENSRecorderDataHandler<T>::add(T&& item)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _queue.push_back(std::move(item));
    lock.unlock();
    _condVar.notify_one();
}

template<typename T>
bool SENSRecorderDataHandler<T>::getErrorMsg(std::string& msg)
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

//explicit instantiation
template class SENSRecorderDataHandler<GpsInfo>;
template class SENSRecorderDataHandler<OrientationInfo>;
template class SENSRecorderDataHandler<FrameInfo>;

//-----------------------------------------------------------------------------
SENSGpsRecorderDataHandler::SENSGpsRecorderDataHandler()
  : SENSRecorderDataHandler("gps")
{
}

void SENSGpsRecorderDataHandler::writeLineToFile(ofstream& file, const GpsInfo& data)
{
    file << std::chrono::time_point_cast<SENSMicroseconds>(data.second).time_since_epoch().count() << " "
         << data.first.latitudeDEG << " "
         << data.first.longitudeDEG << " "
         << data.first.altitudeM << " "
         << data.first.accuracyM << "\n";
}

//-----------------------------------------------------------------------------
SENSOrientationRecorderDataHandler::SENSOrientationRecorderDataHandler()
  : SENSRecorderDataHandler("orientation")
{
}

void SENSOrientationRecorderDataHandler::writeLineToFile(ofstream& file, const OrientationInfo& data)
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

//-----------------------------------------------------------------------------
SENSCameraRecorderDataHandler::SENSCameraRecorderDataHandler()
  : SENSRecorderDataHandler("camera")
{
}

void SENSCameraRecorderDataHandler::writeOnThreadStart()
{
    _frameIndex = 0;
}

void SENSCameraRecorderDataHandler::writeLineToFile(ofstream& file, const FrameInfo& data)
{
    if (data.first.empty())
    {
        SENS_WARN("SENSCameraRecorderDataHandler::writeLineToFile: frame is empty");
        return;
    }

    if (!_videoWriter.isOpened())
    {
        std::string filename = _outputDir + "video.avi";
        _videoWriter.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, data.first.size(), true);
        //_videoWriter.open(filename, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, data.first.size(), true);
        SENS_DEBUG("Opening video for writing: %s", filename.c_str());
    }

    if (!_videoWriter.isOpened())
    {
        SENS_WARN("SENSCameraRecorderDataHandler::writeLineToFile: video writer not opened");
    }
    else
    {
        long long timePt = std::chrono::time_point_cast<SENSMicroseconds>(data.second).time_since_epoch().count();
        //write time and frame index
        file << timePt << " " << _frameIndex << "\n";
        //SENS_DEBUG("SENSCameraRecorderDataHandler: time point for frame %d is %s", _frameIndex, timeStr.c_str());
        _frameIndex++;

        _videoWriter.write(data.first);
    }
}

void SENSCameraRecorderDataHandler::writeOnThreadFinish()
{
    if (_videoWriter.isOpened())
        _videoWriter.release();
}

void SENSCameraRecorderDataHandler::updateConfig(const SENSCameraConfig& config)
{
    //write config to file in directory
    std::string fileName = _outputDir + "cameraConfig.json";
    if (Utils::fileExists(fileName))
        Utils::removeFile(fileName);

    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "deviceId" << config.deviceId;
        fs << "widthPix" << config.streamConfig.widthPix;
        fs << "heightPix" << config.streamConfig.heightPix;
        fs << "focalLengthPix" << config.streamConfig.focalLengthPix;
        fs << "focusMode" << (int)config.focusMode;
        fs << "facing" << (int)config.facing;
    }
}
