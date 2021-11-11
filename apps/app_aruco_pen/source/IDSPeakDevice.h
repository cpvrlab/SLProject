//#############################################################################
//  File:      IDSPeakDevice.h
//  Date:      November 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_IDSPEAKDEVICE_H
#define SRC_IDSPEAKDEVICE_H

#include <peak/peak.hpp>
#include <peak_ipl/peak_ipl.hpp>
#include <peak/converters/peak_buffer_converter_ipl.hpp>

#include <IDSPeakInterface.h>
#include <SL.h>

class IDSPeakDevice
{
    friend class IDSPeakInterface;

private:
    std::shared_ptr<peak::core::Device>     device              = nullptr;
    int                                     _deviceIndex        = 0;
    std::shared_ptr<peak::core::DataStream> dataStream          = nullptr;
    std::shared_ptr<peak::core::NodeMap>    nodeMapRemoteDevice = nullptr;

public:
    IDSPeakDevice();
    IDSPeakDevice(std::shared_ptr<peak::core::Device> device,
                  int                                 deviceIndex);

    void captureImage(int*      width,
                      int*      height,
                      uint8_t** dataBGR,
                      uint8_t** dataGray);
    void close();

    int deviceIndex() { return _deviceIndex; }

private:
    void prepare();
    void setParameters();
    void allocateBuffers();
    void deallocateBuffers();
    void startCapture();
    void stopCapture();
};

#endif // SRC_IDSPEAKDEVICE_H
