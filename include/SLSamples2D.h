//#############################################################################
//  File:      SLSamples2D.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSAMPLES2D_H
#define SLSAMPLES2D_H

#include <SL.h>
#include <SLVec2.h>

//-----------------------------------------------------------------------------
//! Class for 2D disk samplepoints
class SLSamples2D
{  public:  
                        SLSamples2D(){samples(1,1);} 
                       ~SLSamples2D(){}   
            // Setters
            void        samples(SLuint x, SLuint y,
                                SLbool evenlyDistributed=true);
            void        point(SLuint x, SLuint y, SLVec2f point)
                        {  _points[x*_samplesY + y].set(point);
                        }  
            // Getters
            SLuint      samplesX(){return _samplesX;}
            SLuint      samplesY(){return _samplesY;}
            SLuint      samples (){return _samples;}
            SLVec2f     point(SLuint x,SLuint y){return _points[x*_samplesY + y];}
            SLuint      sizeInBytes() {return (SLuint)(_points.size() * sizeof(SLVec2f));}
   private:
            void        distribConcentric (SLbool evenlyDistributed); 
            SLVec2f     mapSquareToDisc   (SLfloat x, SLfloat y);

            SLuint      _samplesX;    //!< No. of samples in x direction
            SLuint      _samplesY;    //!< No. of samples in y direction
            SLuint      _samples;     //!< No. of samples = samplesX x samplesY
            SLVVec2f    _points;      //!< samplepoints for distributed tracing 
};
//-----------------------------------------------------------------------------
#endif


