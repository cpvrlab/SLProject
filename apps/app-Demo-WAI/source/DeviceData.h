#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H
#include <string>

typedef struct AppDirectories
{
    std::string writableDir;
    std::string waiDataRoot;
    std::string slDataRoot;
    std::string vocabularyDir;
    std::string logFileDir;

} AppDirectories;

//! Data that is different on different devices and different plattforms
class DeviceData
{
public:
    explicit DeviceData(int            scrWidth,
                        int            scrHeight,
                        float          scr2fbX,
                        float          scr2fbY,
                        int            dpi,
                        AppDirectories dirs)
    {
        _dpi       = dpi;
        _scrWidth  = (int)(scrWidth * scr2fbX);
        _scrHeight = (int)(scrHeight * scr2fbY);
        _dirs      = dirs;

        _fontDir  = _dirs.slDataRoot + "/images/fonts/";
        _videoDir = _dirs.writableDir + "erleb-AR/locations/";
        _calibDir = _dirs.writableDir + "calibrations/";
        _mapDir   = _dirs.writableDir + "maps/";
    }
    DeviceData() = delete;

    int                   scrWidth() const { return _scrWidth; }
    int                   scrHeight() const { return _scrHeight; }
    int                   dpi() const { return _dpi; }
    const AppDirectories& dirs() const { return _dirs; }
    const std::string&    fontDir() const { return _fontDir; }
    const std::string&    videoDir() const { return _videoDir; }
    const std::string&    calibDir() const { return _calibDir; }
    const std::string&    mapDir() const { return _mapDir; }

private:
    //screen width
    int _scrWidth;
    //screen height
    int _scrHeight;
    //dots per inch of screen
    int _dpi;
    //application directories plattform and device specific
    AppDirectories _dirs;
    //path to fonts
    std::string _fontDir;
    //path to videos
    std::string _videoDir;
    //path to calibrations
    std::string _calibDir;
    //path to maps
    std::string _mapDir;
};

#endif //DEVICE_DATA_H
