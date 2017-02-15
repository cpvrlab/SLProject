//#############################################################################
//  File:      SLCVFeatureMap2D.h
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVFEATUREMAPPER2D_H
#define SLCVFEATUREMAPPER2D_H

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracker
for a good top down information.
*/

#include <SLCV.h>
#include <SLCVFeatureMap2D.h>

using namespace std;

//-----------------------------------------------------------------------------
class SLCVFeatureMapper2D
{
    public:
        enum Mapper2DState {IDLE, LINE_INPUT, CAPTURE};

                SLCVFeatureMapper2D ();

        void    create              (SLCVMat image,
                                     SLfloat offsetXMM,
                                     SLfloat offsetYMM,
                                     SLstring filename,
                                     SLCVFeatureType type);

        void    clear               ();
        bool    stateIsLineInput    () {return _state == LINE_INPUT;}
        bool    stateIsCapture      () {return _state == CAPTURE;}
        bool    stateIsIdle         () {return _state == IDLE;}

        void    addDigit            (string str) {_refWidthStr << str;}
        string  getRefWidthStr      () {return _refWidthStr.str();}
        void    state               (Mapper2DState newState) {_state = newState;}

        void    removeLastDigit ();

    private:
        SLCVMat             _image;         //!< input image
        SLCVFeatureMap2D    _map;           //!< created
        Mapper2DState       _state;         //!< current state
        stringstream        _refWidthStr;   //!< reference width in meter
};
//-----------------------------------------------------------------------------

#endif // SLCVFEATUREMAPPER2D_H
