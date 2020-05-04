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
                        int            dpi,
                        AppDirectories dirs)
    {
        _dpi       = dpi;
        _scrWidth  = scrWidth;
        _scrHeight = scrHeight;
        _dirs      = dirs;

        _fontDir    = _dirs.slDataRoot + "/images/fonts/";
        _textureDir = _dirs.slDataRoot + "/images/textures/";
        _videoDir   = _dirs.writableDir + "erleb-AR/locations/";
        _calibDir   = _dirs.writableDir + "calibrations/";
        _mapDir     = _dirs.writableDir + "maps/";
        _erlebARDir = _dirs.writableDir + "erleb-AR/";
    }
    DeviceData() = delete;

    int                   scrWidth() const { return _scrWidth; }
    int                   scrHeight() const { return _scrHeight; }
    int                   dpi() const { return _dpi; }
    const AppDirectories& dirs() const { return _dirs; }
    const std::string&    fontDir() const { return _fontDir; }
    const std::string&    textureDir() const { return _textureDir; }
    const std::string&    videoDir() const { return _videoDir; }
    const std::string&    calibDir() const { return _calibDir; }
    const std::string&    mapDir() const { return _mapDir; }
    const std::string&    erlebARDir() const { return _erlebARDir; }

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
    //path to textures
    std::string _textureDir;
    //path to videos
    std::string _videoDir;
    //path to calibrations
    std::string _calibDir;
    //path to maps
    std::string _mapDir;
    //erlebar dir
    std::string _erlebARDir;
};

#endif //DEVICE_DATA_H
