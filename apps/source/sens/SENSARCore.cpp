#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <SENSARCore.h>

void SENSARCore::configure(int  targetWidth,
                           int  targetHeight,
                           int  manipWidth,
                           int  manipHeight,
                           bool convertManipToGray)
{
    _config.targetWidth        = targetWidth;
    _config.targetHeight       = targetHeight;
    _config.manipWidth         = targetWidth;
    _config.manipHeight        = manipHeight;
    _config.convertManipToGray = convertManipToGray;
}
