//#############################################################################
//  File:      SLRnd3f.h
//  Author:    Marcus Hudritsch
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLRND3F_H
#define SLRND3F_H

#include <SLVec3.h>

//-----------------------------------------------------------------------------
//! Abstract base class for random 3D point generator
class SLRnd3f
{
    public:
        virtual SLVec3f generate() = 0;
    
    protected:
        default_random_engine _generator;
};
//-----------------------------------------------------------------------------
//! Class for random generator for normal distributed 3D points
class SLRnd3fNormal : public SLRnd3f
{
    public:
        SLRnd3fNormal(SLVec3f mean, SLVec3f stddev)
        {   _xDistribution = new normal_distribution<SLfloat>(mean.x, stddev.x);
            _yDistribution = new normal_distribution<SLfloat>(mean.y, stddev.y);
            _zDistribution = new normal_distribution<SLfloat>(mean.z, stddev.z);
        }

       ~SLRnd3fNormal()
        {   delete _xDistribution;
            delete _yDistribution;
            delete _zDistribution;
        }

        SLVec3f generate()
        {   SLfloat x = _xDistribution->operator()(_generator);
            SLfloat y = _yDistribution->operator()(_generator);
            SLfloat z = _zDistribution->operator()(_generator);
            return SLVec3f(x,y,z);
        }
    
    private: 
        normal_distribution<SLfloat>* _xDistribution;
        normal_distribution<SLfloat>* _yDistribution;
        normal_distribution<SLfloat>* _zDistribution;
};
//-----------------------------------------------------------------------------
//! Class for random generator for uniform distributed 3D points
class SLRnd3fUniform : public SLRnd3f
{
    public:
        SLRnd3fUniform(SLVec3f min, SLVec3f max)
        {   _xDistribution = new uniform_real_distribution<SLfloat>(min.x, max.x);
            _yDistribution = new uniform_real_distribution<SLfloat>(min.y, max.y);
            _zDistribution = new uniform_real_distribution<SLfloat>(min.z, max.z);
        }

       ~SLRnd3fUniform()
        {   delete _xDistribution;
            delete _yDistribution;
            delete _zDistribution;
        }

        SLVec3f generate()
        {   SLfloat x = _xDistribution->operator()(_generator);
            SLfloat y = _yDistribution->operator()(_generator);
            SLfloat z = _zDistribution->operator()(_generator);
            return SLVec3f(x,y,z);
        }
    
    private: 
        uniform_real_distribution<SLfloat>* _xDistribution;
        uniform_real_distribution<SLfloat>* _yDistribution;
        uniform_real_distribution<SLfloat>* _zDistribution;
};
//-----------------------------------------------------------------------------
#endif
