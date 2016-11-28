//#############################################################################
//  File:      AR2DMapper.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVFEATUREMAPPER2D_H
#define SLCVFEATUREMAPPER2D_H

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
