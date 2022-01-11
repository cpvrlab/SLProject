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

#include <SL.h>

//-----------------------------------------------------------------------------
struct IDSPeakDeviceParams
{
    double frameRate = 10.0;
    double gain      = 1.0;
    double gamma     = 1.0;
    int    binning   = 1;
};
//-----------------------------------------------------------------------------
struct IDSPeakFrame
{
    int      width;
    int      height;
    uint8_t* dataBGR;
    uint8_t* dataGray;
};
//-----------------------------------------------------------------------------
class IDSPeakDevice
{
    friend class IDSPeakInterface;

private:
    std::shared_ptr<peak::core::Device>     _device              = nullptr;
    int                                     _deviceIndex         = 0;
    std::shared_ptr<peak::core::DataStream> _dataStream          = nullptr;
    std::shared_ptr<peak::core::NodeMap>    _nodeMapRemoteDevice = nullptr;

    std::shared_ptr<peak::core::nodes::FloatNode> _gainNode  = nullptr;
    std::shared_ptr<peak::core::nodes::FloatNode> _gammaNode = nullptr;

public:
    IDSPeakDevice();
    IDSPeakDevice(std::shared_ptr<peak::core::Device> device,
                  int                                 deviceIndex);

    IDSPeakFrame acquireImage();

    double gain();
    double gamma();
    void   gain(double gain);
    void   gamma(double gamma);

    void close();

    int deviceIndex() { return _deviceIndex; }

private:
    void prepare(IDSPeakDeviceParams& params);
    void setParameters(IDSPeakDeviceParams& params);
    void allocateBuffers();
    void deallocateBuffers();
    void startCapture();
    void stopCapture();
};
//-----------------------------------------------------------------------------
#endif // SRC_IDSPEAKDEVICE_H
