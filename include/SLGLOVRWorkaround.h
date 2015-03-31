//#############################################################################
//  File:      SLOculus.h
//  Purpose:   Wrapper around Oculus Rift
//  Author:    Marc Wacker, Roman Kühne, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLOVRWORKAROUND_H
#define SLOVRWORKAROUND_H

#ifndef SL_OVR

#include <stdafx.h>

//-------------------------------------------------------------------------------------
enum DistortionEqnType
{
    Distortion_No_Override  = -1,    
    // These two are leagcy and deprecated.
    Distortion_Poly4        = 0,    // scale = (K0 + K1*r^2 + K2*r^4 + K3*r^6)
    Distortion_RecipPoly4   = 1,    // scale = 1/(K0 + K1*r^2 + K2*r^4 + K3*r^6)

    // CatmullRom10 is the preferred distortion format.
    Distortion_CatmullRom10 = 2,    // scale = Catmull-Rom spline through points (1.0, K[1]...K[9])

    Distortion_LAST                 // For ease of enumeration.
};

//-------------------------------------------------------------------------------------
// HMD types.
//
enum HmdTypeEnum
{
    HmdType_None,
    HmdType_DKProto,            // First duct-tape model, never sold.
    HmdType_DK1,                // DevKit1 - on sale to developers.
    HmdType_DKHDProto,          // DKHD - shown at various shows, never sold.
    HmdType_DKHD2Proto,         // DKHD2, 5.85-inch panel, never sold.
    HmdType_DKHDProto566Mi,     // DKHD, 5.66-inch panel, never sold.
    HmdType_CrystalCoveProto,   // Crystal Cove, 5.66-inch panel, shown at shows but never sold.
    HmdType_DK2,

    // Reminder - this header file is public - codenames only!
    HmdType_Unknown,            // Used for unnamed HW lab experiments.
    HmdType_LAST
};
//-------------------------------------------------------------------------------------
// HMD shutter types.
//
enum HmdShutterTypeEnum
{
    HmdShutter_Global,
    HmdShutter_RollingTopToBottom,
    HmdShutter_RollingLeftToRight,
    HmdShutter_RollingRightToLeft,
    // TODO:
    // color-sequential e.g. LCOS?
    // alternate eyes?
    // alternate columns?
    // outside-in?

    HmdShutter_LAST
};


//-------------------------------------------------------------------------------------
// For headsets that use eye cups
//
enum EyeCupType
{
    // Public lenses
    EyeCup_DK1A = 0,
    EyeCup_DK1B = 1,
    EyeCup_DK1C = 2,
    EyeCup_DK2A = 3,

    // Internal R&D codenames.
    // Reminder - this header file is public - codenames only!
    EyeCup_DKHD2A,
    EyeCup_OrangeA,
    EyeCup_RedA,
    EyeCup_PinkA,
    EyeCup_BlueA,
    EyeCup_Delilah1A,
    EyeCup_Delilah2A,
    EyeCup_JamesA,
    EyeCup_SunMandalaA,

    EyeCup_LAST
};

//-------------------------------------------------------------------------------------
bool FitCubicPolynomial ( float *pResult, const float *pFitX, const float *pFitY )
{
    float d0 = ((pFitX[0]-pFitX[1]) * (pFitX[0]-pFitX[2]) * (pFitX[0]-pFitX[3]));
    float d1 = ((pFitX[1]-pFitX[2]) * (pFitX[1]-pFitX[3]) * (pFitX[1]-pFitX[0]));
    float d2 = ((pFitX[2]-pFitX[3]) * (pFitX[2]-pFitX[0]) * (pFitX[2]-pFitX[1]));
    float d3 = ((pFitX[3]-pFitX[0]) * (pFitX[3]-pFitX[1]) * (pFitX[3]-pFitX[2]));

    if ( ( d0 == 0.0f ) || ( d1 == 0.0f ) || ( d2 == 0.0f ) || ( d3 == 0.0f ) )
    {
        return false;
    }

    float f0 = pFitY[0] / d0;
    float f1 = pFitY[1] / d1;
    float f2 = pFitY[2] / d2;
    float f3 = pFitY[3] / d3;

    pResult[0] = -( f0*pFitX[1]*pFitX[2]*pFitX[3]
                  + f1*pFitX[0]*pFitX[2]*pFitX[3]
                  + f2*pFitX[0]*pFitX[1]*pFitX[3]
                  + f3*pFitX[0]*pFitX[1]*pFitX[2] );
    pResult[1] = f0*(pFitX[1]*pFitX[2] + pFitX[2]*pFitX[3] + pFitX[3]*pFitX[1])
               + f1*(pFitX[0]*pFitX[2] + pFitX[2]*pFitX[3] + pFitX[3]*pFitX[0])
               + f2*(pFitX[0]*pFitX[1] + pFitX[1]*pFitX[3] + pFitX[3]*pFitX[0])
               + f3*(pFitX[0]*pFitX[1] + pFitX[1]*pFitX[2] + pFitX[2]*pFitX[0]);
    pResult[2] = -( f0*(pFitX[1]+pFitX[2]+pFitX[3])
                  + f1*(pFitX[0]+pFitX[2]+pFitX[3])
                  + f2*(pFitX[0]+pFitX[1]+pFitX[3])
                  + f3*(pFitX[0]+pFitX[1]+pFitX[2]) );
    pResult[3] = f0 + f1 + f2 + f3;

    return true;
}

//-------------------------------------------------------------------------------------
enum { NumCoefficients = 11 };
#define TPH_SPLINE_STATISTICS 0
#if TPH_SPLINE_STATISTICS
static float max_scaledVal = 0;
static float average_total_out_of_range = 0;
static float average_out_of_range;
static int num_total = 0;
static int num_out_of_range = 0;
static int num_out_of_range_over_1 = 0;
static int num_out_of_range_over_2 = 0;
static int num_out_of_range_over_3 = 0;
static float percent_out_of_range;
#endif

//-------------------------------------------------------------------------------------
float EvalCatmullRom10Spline ( float const *K, float scaledVal )
{
    int const NumSegments = NumCoefficients;

#if TPH_SPLINE_STATISTICS
    //Value should be in range of 0 to (NumSegments-1) (typically 10) if spline is valid. Right?
    if (scaledVal > (NumSegments-1))
    {
        num_out_of_range++;
        average_total_out_of_range+=scaledVal;
        average_out_of_range = average_total_out_of_range / ((float) num_out_of_range); 
        percent_out_of_range = 100.0f*(num_out_of_range)/num_total;
    }
    if (scaledVal > (NumSegments-1+1)) num_out_of_range_over_1++;
    if (scaledVal > (NumSegments-1+2)) num_out_of_range_over_2++;
    if (scaledVal > (NumSegments-1+3)) num_out_of_range_over_3++;
    num_total++;
    if (scaledVal > max_scaledVal)
    {
        max_scaledVal = scaledVal;
        max_scaledVal = scaledVal;
    }
#endif

    float scaledValFloor = floorf(scaledVal);
    scaledValFloor = SL_max(0.0f, SL_min((float)(NumSegments - 1), scaledValFloor));
    float t = scaledVal - scaledValFloor;
    int k = (int)scaledValFloor;

    float p0, p1;
    float m0, m1;
    switch(k)
    {
    case 0:
        // Curve starts at 1.0 with gradient K[1]-K[0]
        p0 = 1.0f;
        m0 = (K[1] - K[0]);    // general case would have been (K[1]-K[-1])/2
        p1 = K[1];
        m1 = 0.5f * (K[2] - K[0]);
        break;
    default:
        // General case
        p0 = K[k];
        m0 = 0.5f * (K[k + 1] - K[k - 1]);
        p1 = K[k + 1];
        m1 = 0.5f * (K[k + 2] - K[k]);
        break;
    case NumSegments - 2:
        // Last tangent is just the slope of the last two points.
        p0 = K[NumSegments - 2];
        m0 = 0.5f * (K[NumSegments - 1] - K[NumSegments - 2]);
        p1 = K[NumSegments - 1];
        m1 = K[NumSegments - 1] - K[NumSegments - 2];
        break;
    case NumSegments - 1:
        // Beyond the last segment it's just a straight line
        p0 = K[NumSegments - 1];
        m0 = K[NumSegments - 1] - K[NumSegments - 2];
        p1 = p0 + m0;
        m1 = m0;
        break;
    }

    float omt = 1.0f - t;
    float res = (p0 * (1.0f + 2.0f *   t) + m0 *   t) * omt * omt
        + (p1 * (1.0f + 2.0f * omt) - m1 * omt) *   t *   t;

    return res;
}

//-------------------------------------------------------------------------------------
struct LensConfig
{
    // The result is a scaling applied to the distance from the center of the lens.
    float    DistortionFnScaleRadiusSquared (float rsq) const
    {
        float scale = 1.0f;
        switch ( Eqn )
        {
        case Distortion_Poly4:
            // This version is deprecated! Prefer one of the other two.
            scale = ( K[0] + rsq * ( K[1] + rsq * ( K[2] + rsq * K[3] ) ) );
            break;
        case Distortion_RecipPoly4:
            scale = 1.0f / ( K[0] + rsq * ( K[1] + rsq * ( K[2] + rsq * K[3] ) ) );
            break;
        case Distortion_CatmullRom10:{
            // A Catmull-Rom spline through the values 1.0, K[1], K[2] ... K[10]
            // evenly spaced in R^2 from 0.0 to MaxR^2
            // K[0] controls the slope at radius=0.0, rather than the actual value.
            const int NumSegments = NumCoefficients;
            assert ( NumSegments <= NumCoefficients );
            float scaledRsq = (float)(NumSegments-1) * rsq / ( MaxR * MaxR );
            scale = EvalCatmullRom10Spline ( K, scaledRsq );


            }break;
        default:
            assert ( false );
            break;
        }
        return scale;
    }
    // x,y,z components map to r,g,b scales.
    SLVec3f DistortionFnScaleRadiusSquaredChroma (float rsq) const
    {
        float scale = DistortionFnScaleRadiusSquared ( rsq );
        SLVec3f scaleRGB;
        scaleRGB.x = scale * ( 1.0f + ChromaticAberration[0] + rsq * ChromaticAberration[1] );     // Red
        scaleRGB.y = scale;                                                                        // Green
        scaleRGB.z = scale * ( 1.0f + ChromaticAberration[2] + rsq * ChromaticAberration[3] );     // Blue
        return scaleRGB;
    }

    // DistortionFn applies distortion to the argument.
    // Input: the distance in TanAngle/NIC space from the optical center to the input pixel.
    // Output: the resulting distance after distortion.
    float DistortionFn(float r) const
    {
        return r * DistortionFnScaleRadiusSquared ( r * r );
    }

    // DistortionFnInverse computes the inverse of the distortion function on an argument.
    float DistortionFnInverse(float r) const
    {    
        assert((r <= 20.0f));

        float s, d;
        float delta = r * 0.25f;

        // Better to start guessing too low & take longer to converge than too high
        // and hit singularities. Empirically, r * 0.5f is too high in some cases.
        s = r * 0.25f;
        d = fabs(r - DistortionFn(s));

        for (int i = 0; i < 20; i++)
        {
            float sUp   = s + delta;
            float sDown = s - delta;
            float dUp   = fabs(r - DistortionFn(sUp));
            float dDown = fabs(r - DistortionFn(sDown));

            if (dUp < d)
            {
                s = sUp;
                d = dUp;
            }
            else if (dDown < d)
            {
                s = sDown;
                d = dDown;
            }
            else
            {
                delta *= 0.5f;
            }
        }

        return s;
    }
    

    // Also computes the inverse, but using a polynomial approximation. Warning - it's just an approximation!
    float DistortionFnInverseApprox(float r) const
    {
        float rsq = r * r;
        float scale = 1.0f;
        switch ( Eqn )
        {
        case Distortion_Poly4:
            // Deprecated
            assert ( false );
            break;
        case Distortion_RecipPoly4:
            scale = 1.0f / ( InvK[0] + rsq * ( InvK[1] + rsq * ( InvK[2] + rsq * InvK[3] ) ) );
            break;
        case Distortion_CatmullRom10:{
            // A Catmull-Rom spline through the values 1.0, K[1], K[2] ... K[9]
            // evenly spaced in R^2 from 0.0 to MaxR^2
            // K[0] controls the slope at radius=0.0, rather than the actual value.
            const int NumSegments = NumCoefficients;
            assert ( NumSegments <= NumCoefficients );
            float scaledRsq = (float)(NumSegments-1) * rsq / ( MaxInvR * MaxInvR );
            scale = EvalCatmullRom10Spline ( InvK, scaledRsq );


            }break;
        default:
            assert ( false );
            break;
        }
        return r * scale;
        }
        // Sets up InvK[].
        void SetUpInverseApprox()
        {
            float maxR = MaxInvR;

            switch ( Eqn )
            {
            case Distortion_Poly4:
                // Deprecated
                assert ( false );
                break;
            case Distortion_RecipPoly4:{

                float sampleR[4];
                float sampleRSq[4];
                float sampleInv[4];
                float sampleFit[4];

                // Found heuristically...
                sampleR[0] = 0.0f;
                sampleR[1] = maxR * 0.4f;
                sampleR[2] = maxR * 0.8f;
                sampleR[3] = maxR * 1.5f;
                for ( int i = 0; i < 4; i++ )
                {
                    sampleRSq[i] = sampleR[i] * sampleR[i];
                    sampleInv[i] = DistortionFnInverse ( sampleR[i] );
                    sampleFit[i] = sampleR[i] / sampleInv[i];
                }
                sampleFit[0] = 1.0f;
                FitCubicPolynomial ( InvK, sampleRSq, sampleFit );

            #if 0
                // Should be a nearly exact match on the chosen points.
                OVR_ASSERT ( fabs ( DistortionFnInverse ( sampleR[0] ) - DistortionFnInverseApprox ( sampleR[0] ) ) / maxR < 0.0001f );
                OVR_ASSERT ( fabs ( DistortionFnInverse ( sampleR[1] ) - DistortionFnInverseApprox ( sampleR[1] ) ) / maxR < 0.0001f );
                OVR_ASSERT ( fabs ( DistortionFnInverse ( sampleR[2] ) - DistortionFnInverseApprox ( sampleR[2] ) ) / maxR < 0.0001f );
                OVR_ASSERT ( fabs ( DistortionFnInverse ( sampleR[3] ) - DistortionFnInverseApprox ( sampleR[3] ) ) / maxR < 0.0001f );
                // Should be a decent match on the rest of the range.
                const int maxCheck = 20;
                for ( int i = 0; i < maxCheck; i++ )
                {
                    float checkR = (float)i * maxR / (float)maxCheck;
                    float realInv = DistortionFnInverse       ( checkR );
                    float testInv = DistortionFnInverseApprox ( checkR );
                    float error = fabsf ( realInv - testInv ) / maxR;
                    OVR_ASSERT ( error < 0.1f );
                }
            #endif

                }break;
            case Distortion_CatmullRom10:{

                const int NumSegments = NumCoefficients;
                assert ( NumSegments <= NumCoefficients );
                for ( int i = 1; i < NumSegments; i++ )
                {
                    float scaledRsq = (float)i;
                    float rsq = scaledRsq * MaxInvR * MaxInvR / (float)( NumSegments - 1);
                    float r = sqrtf ( rsq );
                    float inv = DistortionFnInverse ( r );
                    InvK[i] = inv / r;
                    InvK[0] = 1.0f;     // TODO: fix this.
                }

        #if 0
                const int maxCheck = 20;
                for ( int i = 0; i <= maxCheck; i++ )
                {
                    float checkR = (float)i * MaxInvR / (float)maxCheck;
                    float realInv = DistortionFnInverse       ( checkR );
                    float testInv = DistortionFnInverseApprox ( checkR );
                    float error = fabsf ( realInv - testInv ) / MaxR;
                    OVR_ASSERT ( error < 0.01f );
                }
        #endif

                }break;

            default:
                break;
            }
        }

    // Sets a bunch of sensible defaults.
    void SetToIdentity()
    {
        for ( int i = 0; i < NumCoefficients; i++ )
        {
            K[i] = 0.0f;
            InvK[i] = 0.0f;
        }
        Eqn = Distortion_RecipPoly4;
        K[0] = 1.0f;
        InvK[0] = 1.0f;
        MaxR = 1.0f;
        MaxInvR = 1.0f;
        ChromaticAberration[0] = 0.0f;
        ChromaticAberration[1] = 0.0f;
        ChromaticAberration[2] = 0.0f;
        ChromaticAberration[3] = 0.0f;
        MetersPerTanAngleAtCenter = 0.05f;
    }



    DistortionEqnType   Eqn;
    float               K[NumCoefficients];
    float               MaxR;       // The highest R you're going to query for - the curve is unpredictable beyond it.

    float               MetersPerTanAngleAtCenter;

    // Additional per-channel scaling is applied after distortion:
    //  Index [0] - Red channel constant coefficient.
    //  Index [1] - Red channel r^2 coefficient.
    //  Index [2] - Blue channel constant coefficient.
    //  Index [3] - Blue channel r^2 coefficient.
    float               ChromaticAberration[4];

    float               InvK[NumCoefficients];
    float               MaxInvR;
};

//-------------------------------------------------------------------------------------
struct ovrSizei {
    int w, h;
};
//-------------------------------------------------------------------------------------
struct ovrSizef {
    float w, h;
};
//-------------------------------------------------------------------------------------
struct HmdRenderInfo
{
    // The start of this structure is intentionally very similar to HMDInfo in OVER_Device.h
    // However to reduce interdependencies, one does not simply #include the other.

    HmdTypeEnum HmdType;

    // Size of the entire screen
    ovrSizei    ResolutionInPixels;
    ovrSizef    ScreenSizeInMeters;
    float       ScreenGapSizeInMeters;

    // Characteristics of the lenses.
    float       CenterFromTopInMeters;
    float       LensSeparationInMeters;
    float       LensDiameterInMeters;
    float       LensSurfaceToMidplateInMeters;
    EyeCupType  EyeCups;

    // Timing & shutter data. All values in seconds.
    struct ShutterInfo
    {
        HmdShutterTypeEnum  Type;
        float               VsyncToNextVsync;                // 1/framerate
        float               VsyncToFirstScanline;            // for global shutter, vsync->shutter open.
        float               FirstScanlineToLastScanline;     // for global shutter, will be zero.
        float               PixelSettleTime;                 // estimated.
        float               PixelPersistence;                // Full persistence = 1/framerate.
    }           Shutter;


    // These are all set from the user's profile.
    struct EyeConfig
    {
        // Distance from center of eyeball to front plane of lens.
        float               ReliefInMeters;
        // Distance from nose (technically, center of Rift) to the middle of the eye.
        float               NoseToPupilInMeters;

        LensConfig          Distortion;
    } EyeLeft, EyeRight;


    HmdRenderInfo()
    {
        HmdType = HmdType_None;
        ScreenGapSizeInMeters = 0.0f;
        CenterFromTopInMeters = 0.0f;
        LensSeparationInMeters = 0.0f;
        LensDiameterInMeters = 0.0f;
        LensSurfaceToMidplateInMeters = 0.0f;
        Shutter.Type = HmdShutter_LAST;
        Shutter.VsyncToNextVsync = 0.0f;
        Shutter.VsyncToFirstScanline = 0.0f;
        Shutter.FirstScanlineToLastScanline = 0.0f;
        Shutter.PixelSettleTime = 0.0f;
        Shutter.PixelPersistence = 0.0f;
        EyeCups = EyeCup_DK1A;
        EyeLeft.ReliefInMeters = 0.0f;
        EyeLeft.NoseToPupilInMeters = 0.0f;
        EyeLeft.Distortion.SetToIdentity();
        EyeRight = EyeLeft;
    }

    // The "center eye" is the position the HMD tracking returns,
    // and games will also usually use it for audio, aiming reticles, some line-of-sight tests, etc.
    EyeConfig GetEyeCenter() const
    {
        EyeConfig result;
        result.ReliefInMeters = 0.5f * ( EyeLeft.ReliefInMeters + EyeRight.ReliefInMeters );
        result.NoseToPupilInMeters = 0.0f;
        result.Distortion.SetToIdentity();
        return result;
    }

};

//-------------------------------------------------------------------------------------
SLMat4f ovrMatrix4f_OrthoSubProjection(SLMat4f projection, SLVec2f orthoScale,
                                       float orthoDistance, float eyeViewAdjustX)
{

    float orthoHorizontalOffset = eyeViewAdjustX / orthoDistance;

    // Current projection maps real-world vector (x,y,1) to the RT.
    // We want to find the projection that maps the range [-FovPixels/2,FovPixels/2] to
    // the physical [-orthoHalfFov,orthoHalfFov]
    // Note moving the offset from M[0][2]+M[1][2] to M[0][3]+M[1][3] - this means
    // we don't have to feed in Z=1 all the time.
    // The horizontal offset math is a little hinky because the destination is
    // actually [-orthoHalfFov+orthoHorizontalOffset,orthoHalfFov+orthoHorizontalOffset]
    // So we need to first map [-FovPixels/2,FovPixels/2] to
    //                         [-orthoHalfFov+orthoHorizontalOffset,orthoHalfFov+orthoHorizontalOffset]:
    // x1 = x0 * orthoHalfFov/(FovPixels/2) + orthoHorizontalOffset;
    //    = x0 * 2*orthoHalfFov/FovPixels + orthoHorizontalOffset;
    // But then we need the sam mapping as the existing projection matrix, i.e.
    // x2 = x1 * Projection.M[0][0] + Projection.M[0][2];
    //    = x0 * (2*orthoHalfFov/FovPixels + orthoHorizontalOffset) * Projection.M[0][0] + Projection.M[0][2];
    //    = x0 * Projection.M[0][0]*2*orthoHalfFov/FovPixels +
    //      orthoHorizontalOffset*Projection.M[0][0] + Projection.M[0][2];
    // So in the new projection matrix we need to scale by Projection.M[0][0]*2*orthoHalfFov/FovPixels and
    // offset by orthoHorizontalOffset*Projection.M[0][0] + Projection.M[0][2].

    SLfloat orthoData[16] = { 1.0f, 0.0f, 0.0f, 0.0f, 
                                0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 1.0f};

    orthoData[0] = projection.m(0) * orthoScale.x;
    orthoData[4] = 0.0f;
    orthoData[8] = 0.0f;
    orthoData[12] = -projection.m(8) + ( orthoHorizontalOffset * projection.m(0) );

    orthoData[1] = 0.0f;
    orthoData[5] = -projection.m(5) * orthoScale.y;       // Note sign flip (text rendering uses Y=down).
    orthoData[9] = 0.0f;
    orthoData[13] = -projection.m(9);

    /*
    if ( fabsf ( zNear - zFar ) < 0.001f )
    {
        orthoData[2][0] = 0.0f;
        orthoData[2][1] = 0.0f;
        orthoData[2][2] = 0.0f;
        orthoData[2][3] = zFar;
    }
    else
    {
        orthoData[2][0] = 0.0f;
        orthoData[2][1] = 0.0f;
        orthoData[2][2] = zFar / (zNear - zFar);
        orthoData[2][3] = (zFar * zNear) / (zNear - zFar);
    }
    */

    // mA: Undo effect of sign
    orthoData[2] = 0.0f;
    orthoData[1] = 0.0f;
    //orthoData[2][2] = projection.m[2][2] * projection.m[3][2] * -1.0f; // reverse right-handedness
    orthoData[10] = 0.0f;
    orthoData[14] = 0.0f;
        //projection.m[2][3];

    // No perspective correction for ortho.
    orthoData[3] = 0.0f;
    orthoData[7] = 0.0f;
    orthoData[11] = 0.0f;
    orthoData[15] = 1.0f;
    
    SLMat4f ortho(orthoData);


    return ortho;
}

//-------------------------------------------------------------------------------------
struct DistortionRenderDesc
{
    // The raw lens values.
    LensConfig          Lens;

    // These map from [-1,1] across the eye being rendered into TanEyeAngle space (but still distorted)
    SLVec2f            LensCenter;
    SLVec2f            TanEyeAngleScale;
    // Computed from device characteristics, IPD and eye-relief.
    // (not directly used for rendering, but very useful)
    SLVec2f            PixelsPerTanAngleAtCenter;
};

//-------------------------------------------------------------------------------------
typedef struct ovrDistortionVertex_
{
    SLVec2f ScreenPosNDC;    // [-1,+1],[-1,+1] over the entire framebuffer.
    float       TimeWarpFactor;  // Lerp factor between time-warp matrices. Can be encoded in Pos.z.
    float       VignetteFactor;  // Vignette fade factor. Can be encoded in Pos.w.
    SLVec2f TanEyeAnglesR;
    SLVec2f TanEyeAnglesG;
    SLVec2f TanEyeAnglesB;    
} ovrDistortionVertex;

//-------------------------------------------------------------------------------------
typedef struct ovrDistortionMesh_
{
    ovrDistortionVertex* pVertexData;
    unsigned short*      pIndexData;
    unsigned int         VertexCount;
    unsigned int         IndexCount;
} ovrDistortionMesh;

//-------------------------------------------------------------------------------------
struct DistortionMeshVertexData
{
    // [-1,+1],[-1,+1] over the entire framebuffer.
    SLVec2f    ScreenPosNDC;
    // [0.0-1.0] interpolation value for timewarping - see documentation for details.
    float       TimewarpLerp;
    // [0.0-1.0] fade-to-black at the edges to reduce peripheral vision noise.
    float       Shade;        
    // The red, green, and blue vectors in tan(angle) space.
    // Scale and offset by the values in StereoEyeParams.EyeToSourceUV.Scale
    // and StereoParams.EyeToSourceUV.Offset to get to real texture UV coords.
    SLVec2f    TanEyeAnglesR;
    SLVec2f    TanEyeAnglesG;
    SLVec2f    TanEyeAnglesB;    
};

//-----------------------------------------------------------------------------------
// A set of "reverse-mapping" functions, mapping from real-world and/or texture space back to the framebuffer.

SLVec2f TransformTanFovSpaceToScreenNDC( DistortionRenderDesc const &distortion,
                                          const SLVec2f &tanEyeAngle, bool usePolyApprox /*= false*/ )
{
    float tanEyeAngleRadius = tanEyeAngle.length();
    float tanEyeAngleDistortedRadius = distortion.Lens.DistortionFnInverseApprox ( tanEyeAngleRadius );
    if ( !usePolyApprox )
    {
        tanEyeAngleDistortedRadius = distortion.Lens.DistortionFnInverse ( tanEyeAngleRadius );
    }
    SLVec2f tanEyeAngleDistorted = tanEyeAngle;
    if ( tanEyeAngleRadius > 0.0f )
    {   
        tanEyeAngleDistorted = tanEyeAngle * ( tanEyeAngleDistortedRadius / tanEyeAngleRadius );
    }

    SLVec2f framebufferNDC;
    framebufferNDC.x = ( tanEyeAngleDistorted.x / distortion.TanEyeAngleScale.x ) + distortion.LensCenter.x;
    framebufferNDC.y = ( tanEyeAngleDistorted.y / distortion.TanEyeAngleScale.y ) + distortion.LensCenter.y;

    return framebufferNDC;
}

//-------------------------------------------------------------------------------------
// Same, with chromatic aberration correction.
void TransformScreenNDCToTanFovSpaceChroma ( SLVec2f *resultR, SLVec2f *resultG, SLVec2f *resultB, 
                                             DistortionRenderDesc const &distortion,
                                             const SLVec2f &framebufferNDC )
{
    // Scale to TanHalfFov space, but still distorted.
    SLVec2f tanEyeAngleDistorted;
    tanEyeAngleDistorted.x = ( framebufferNDC.x - distortion.LensCenter.x ) * distortion.TanEyeAngleScale.x;
    tanEyeAngleDistorted.y = ( framebufferNDC.y - distortion.LensCenter.y ) * distortion.TanEyeAngleScale.y;
    // Distort.
    float radiusSquared = ( tanEyeAngleDistorted.x * tanEyeAngleDistorted.x )
                        + ( tanEyeAngleDistorted.y * tanEyeAngleDistorted.y );
    SLVec3f distortionScales = distortion.Lens.DistortionFnScaleRadiusSquaredChroma ( radiusSquared );
    *resultR = tanEyeAngleDistorted * distortionScales.x;
    *resultG = tanEyeAngleDistorted * distortionScales.y;
    *resultB = tanEyeAngleDistorted * distortionScales.z;
}

//-------------------------------------------------------------------------------------
typedef struct ovrFovPort_
{
    /// The tangent of the angle between the viewing vector and the top edge of the field of view.
    float UpTan;
    /// The tangent of the angle between the viewing vector and the bottom edge of the field of view.
    float DownTan;
    /// The tangent of the angle between the viewing vector and the left edge of the field of view.
    float LeftTan;
    /// The tangent of the angle between the viewing vector and the right edge of the field of view.
    float RightTan;
} ovrFovPort;

//-------------------------------------------------------------------------------------
struct ScaleAndOffset2D
{
    SLVec2f Scale;
    SLVec2f Offset;

    ScaleAndOffset2D(float sx = 0.0f, float sy = 0.0f, float ox = 0.0f, float oy = 0.0f)
      : Scale(sx, sy), Offset(ox, oy)        
    { }
};

//-------------------------------------------------------------------------------------
SLVec2f TransformTanFovSpaceToRendertargetNDC( ScaleAndOffset2D const &eyeToSourceNDC,
                                                SLVec2f const &tanEyeAngle )
{
    SLVec2f textureNDC;
    textureNDC.x = tanEyeAngle.x * eyeToSourceNDC.Scale.x + eyeToSourceNDC.Offset.x;
    textureNDC.y = tanEyeAngle.y * eyeToSourceNDC.Scale.y + eyeToSourceNDC.Offset.y;
    return textureNDC;
}

//-------------------------------------------------------------------------------------
SLVec2f TransformRendertargetNDCToTanFovSpace( const ScaleAndOffset2D &eyeToSourceNDC,
                                                const SLVec2f &textureNDC )
{
    SLVec2f tanEyeAngle;
    tanEyeAngle.x = (textureNDC.x - eyeToSourceNDC.Offset.x) / eyeToSourceNDC.Scale.x;
    tanEyeAngle.y = (textureNDC.y - eyeToSourceNDC.Offset.y) / eyeToSourceNDC.Scale.y;
    return tanEyeAngle;
}

//-------------------------------------------------------------------------------------
ScaleAndOffset2D CreateNDCScaleAndOffsetFromFov ( ovrFovPort tanHalfFov )
{
    float projXScale = 2.0f / ( tanHalfFov.LeftTan + tanHalfFov.RightTan );
    float projXOffset = ( tanHalfFov.LeftTan - tanHalfFov.RightTan ) * projXScale * 0.5f;
    float projYScale = 2.0f / ( tanHalfFov.UpTan + tanHalfFov.DownTan );
    float projYOffset = ( tanHalfFov.UpTan - tanHalfFov.DownTan ) * projYScale * 0.5f;

    ScaleAndOffset2D result;
    result.Scale    = SLVec2f(projXScale, projYScale);
    result.Offset   = SLVec2f(projXOffset, projYOffset);
    // Hey - why is that Y.Offset negated?
    // It's because a projection matrix transforms from world coords with Y=up,
    // whereas this is from NDC which is Y=down.

    return result;
}

//-------------------------------------------------------------------------------------
SLMat4f CreateProjection( bool rightHanded, ovrFovPort tanHalfFov,
                            float zNear /*= 0.01f*/, float zFar /*= 10000.0f*/ )
{
    // A projection matrix is very like a scaling from NDC, so we can start with that.
    ScaleAndOffset2D scaleAndOffset = CreateNDCScaleAndOffsetFromFov ( tanHalfFov );

    float handednessScale = 1.0f;
    if ( rightHanded )
    {
        handednessScale = -1.0f;
    }

    float proj[4][4] = {{1, 0, 0, 0},
                        {0, 1, 0, 0},
                        {0, 0, 1, 0},
                        {0, 0, 0, 1}};

    // Produces X result, mapping clip edges to [-w,+w]
    proj[0][0] = scaleAndOffset.Scale.x;
    proj[0][1] = 0.0f;
    proj[0][2] = handednessScale * scaleAndOffset.Offset.x;
    proj[0][3] = 0.0f;

    // Produces Y result, mapping clip edges to [-w,+w]
    // Hey - why is that YOffset negated?
    // It's because a projection matrix transforms from world coords with Y=up,
    // whereas this is derived from an NDC scaling, which is Y=down.
    proj[1][0] = 0.0f;
    proj[1][1] = scaleAndOffset.Scale.y;
    proj[1][2] = handednessScale * -scaleAndOffset.Offset.y;
    proj[1][3] = 0.0f;

    // Produces Z-buffer result - app needs to fill this in with whatever Z range it wants.
    // We'll just use some defaults for now.
    proj[2][0] = 0.0f;
    proj[2][1] = 0.0f;
    proj[2][2] = -handednessScale * zFar / (zNear - zFar);
    proj[2][3] = (zFar * zNear) / (zNear - zFar);

    // Produces W result (= Z in)
    proj[3][0] = 0.0f;
    proj[3][1] = 0.0f;
    proj[3][2] = handednessScale;
    proj[3][3] = 0.0f;

    SLMat4f projection((SLfloat*)proj);
    projection.transpose();

    return projection;
}

//-------------------------------------------------------------------------------------
void createSLDistortionMesh( DistortionMeshVertexData **ppVertices, uint16_t **ppTriangleListIndices,
                           int *pNumVertices, int *pNumTriangles,
                           bool rightEye,
                           const HmdRenderInfo &hmdRenderInfo, 
                           const DistortionRenderDesc &distortion, const ScaleAndOffset2D &eyeToSourceNDC )
{    

    
static const int DMA_GridSizeLog2   = 6;
static const int DMA_GridSize       = 1<<DMA_GridSizeLog2;
static const int DMA_NumVertsPerEye = (DMA_GridSize+1)*(DMA_GridSize+1);
static const int DMA_NumTrisPerEye  = (DMA_GridSize)*(DMA_GridSize)*2;

    // When does the fade-to-black edge start? Chosen heuristically.
    const float fadeOutBorderFraction = 0.075f;

    // Populate vertex buffer info
    float xOffset = 0.0f;
    float uOffset = 0.0f;

    if (rightEye)
    {
        xOffset = 1.0f;
        uOffset = 0.5f;
    }
    *pNumVertices  = DMA_NumVertsPerEye;
    *pNumTriangles = DMA_NumTrisPerEye;

    *ppVertices = (DistortionMeshVertexData*)(new DistortionMeshVertexData[sizeof(DistortionMeshVertexData)* (*pNumVertices)]);
    *ppTriangleListIndices = (uint16_t*)(new uint16_t[sizeof(uint16_t) * (*pNumTriangles) * 3]);

    // First pass - build up raw vertex data.
    DistortionMeshVertexData* pcurVert = *ppVertices;

    for ( int y = 0; y <= DMA_GridSize; y++ )
    {
        for ( int x = 0; x <= DMA_GridSize; x++ )
        {

            SLVec2f sourceCoordNDC;
            // NDC texture coords [-1,+1]
            sourceCoordNDC.x = 2.0f * ( (float)x / (float)DMA_GridSize ) - 1.0f;
            sourceCoordNDC.y = 2.0f * ( (float)y / (float)DMA_GridSize ) - 1.0f;
            SLVec2f tanEyeAngle = TransformRendertargetNDCToTanFovSpace ( eyeToSourceNDC, sourceCoordNDC );

            // Find a corresponding screen position.
            // Note - this function does not have to be precise - we're just trying to match the mesh tessellation
            // with the shape of the distortion to minimise the number of trianlges needed.
            SLVec2f screenNDC = TransformTanFovSpaceToScreenNDC ( distortion, tanEyeAngle, false );
            // ...but don't let verts overlap to the other eye.
            screenNDC.x = SL_max ( -1.0f, SL_min ( screenNDC.x, 1.0f ) );
            screenNDC.y = SL_max ( -1.0f, SL_min ( screenNDC.y, 1.0f ) );

            // From those screen positions, we then need (effectively) RGB UVs.
            // This is the function that actually matters when doing the distortion calculation.
            SLVec2f tanEyeAnglesR, tanEyeAnglesG, tanEyeAnglesB;
            TransformScreenNDCToTanFovSpaceChroma ( &tanEyeAnglesR, &tanEyeAnglesG, &tanEyeAnglesB,
                                                    distortion, screenNDC );
            
            pcurVert->TanEyeAnglesR = tanEyeAnglesR;
            pcurVert->TanEyeAnglesG = tanEyeAnglesG;
            pcurVert->TanEyeAnglesB = tanEyeAnglesB;
            
            HmdShutterTypeEnum shutterType = hmdRenderInfo.Shutter.Type;
            switch ( shutterType )
            {
            case HmdShutter_Global:
                pcurVert->TimewarpLerp = 0.0f;
                break;
            case HmdShutter_RollingLeftToRight:
                // Retrace is left to right - left eye goes 0.0 -> 0.5, then right goes 0.5 -> 1.0
                pcurVert->TimewarpLerp = screenNDC.x * 0.25f + 0.25f;
                if (rightEye)
                {
                    pcurVert->TimewarpLerp += 0.5f;
                }
                break;
            case HmdShutter_RollingRightToLeft:
                // Retrace is right to left - right eye goes 0.0 -> 0.5, then left goes 0.5 -> 1.0
                pcurVert->TimewarpLerp = 0.75f - screenNDC.x * 0.25f;
                if (rightEye)
                {
                    pcurVert->TimewarpLerp -= 0.5f;
                }
                break;
            case HmdShutter_RollingTopToBottom:
                // Retrace is top to bottom on both eyes at the same time.
                pcurVert->TimewarpLerp = screenNDC.y * 0.5f + 0.5f;
                break;
            default: assert(false); break;
            }

            // Fade out at texture edges.
            // The furthest out will be the blue channel, because of chromatic aberration (true of any standard lens)
            SLVec2f sourceTexCoordBlueNDC = TransformTanFovSpaceToRendertargetNDC ( eyeToSourceNDC, tanEyeAnglesB );
            float edgeFadeIn       = ( 1.0f / fadeOutBorderFraction ) *
                                     ( 1.0f - SL_max ( SL_abs ( sourceTexCoordBlueNDC.x ), SL_abs ( sourceTexCoordBlueNDC.y ) ) );
            // Also fade out at screen edges.
            float edgeFadeInScreen = ( 2.0f / fadeOutBorderFraction ) *
                                     ( 1.0f - SL_max ( SL_abs ( screenNDC.x ), SL_abs ( screenNDC.y ) ) );
            edgeFadeIn = SL_min ( edgeFadeInScreen, edgeFadeIn );

            pcurVert->Shade = SL_max ( 0.0f, SL_min ( edgeFadeIn, 1.0f ) );
            pcurVert->ScreenPosNDC.x = 0.5f * screenNDC.x - 0.5f + xOffset;
            pcurVert->ScreenPosNDC.y = -screenNDC.y;

            pcurVert++;
        }
    }


    // Populate index buffer info  
    uint16_t *pcurIndex = *ppTriangleListIndices;

    for ( int triNum = 0; triNum < DMA_GridSize * DMA_GridSize; triNum++ )
    {
        // Use a Morton order to help locality of FB, texture and vertex cache.
        // (0.325ms raster order -> 0.257ms Morton order)
        assert ( DMA_GridSize <= 256 );
        int x = ( ( triNum & 0x0001 ) >> 0 ) |
                ( ( triNum & 0x0004 ) >> 1 ) |
                ( ( triNum & 0x0010 ) >> 2 ) |
                ( ( triNum & 0x0040 ) >> 3 ) |
                ( ( triNum & 0x0100 ) >> 4 ) |
                ( ( triNum & 0x0400 ) >> 5 ) |
                ( ( triNum & 0x1000 ) >> 6 ) |
                ( ( triNum & 0x4000 ) >> 7 );
        int y = ( ( triNum & 0x0002 ) >> 1 ) |
                ( ( triNum & 0x0008 ) >> 2 ) |
                ( ( triNum & 0x0020 ) >> 3 ) |
                ( ( triNum & 0x0080 ) >> 4 ) |
                ( ( triNum & 0x0200 ) >> 5 ) |
                ( ( triNum & 0x0800 ) >> 6 ) |
                ( ( triNum & 0x2000 ) >> 7 ) |
                ( ( triNum & 0x8000 ) >> 8 );
        int FirstVertex = x * (DMA_GridSize+1) + y;
        // Another twist - we want the top-left and bottom-right quadrants to
        // have the triangles split one way, the other two split the other.
        // +---+---+---+---+
        // |  /|  /|\  |\  |
        // | / | / | \ | \ |
        // |/  |/  |  \|  \|
        // +---+---+---+---+
        // |  /|  /|\  |\  |
        // | / | / | \ | \ |
        // |/  |/  |  \|  \|
        // +---+---+---+---+
        // |\  |\  |  /|  /|
        // | \ | \ | / | / |
        // |  \|  \|/  |/  |
        // +---+---+---+---+
        // |\  |\  |  /|  /|
        // | \ | \ | / | / |
        // |  \|  \|/  |/  |
        // +---+---+---+---+
        // This way triangle edges don't span long distances over the distortion function,
        // so linear interpolation works better & we can use fewer tris.
        if ( ( x < DMA_GridSize/2 ) != ( y < DMA_GridSize/2 ) )       // != is logical XOR
        {
            *pcurIndex++ = (uint16_t)FirstVertex;
            *pcurIndex++ = (uint16_t)FirstVertex+1;
            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1)+1;

            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1)+1;
            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1);
            *pcurIndex++ = (uint16_t)FirstVertex;
        }
        else
        {
            *pcurIndex++ = (uint16_t)FirstVertex;
            *pcurIndex++ = (uint16_t)FirstVertex+1;
            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1);

            *pcurIndex++ = (uint16_t)FirstVertex+1;
            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1)+1;
            *pcurIndex++ = (uint16_t)FirstVertex+(DMA_GridSize+1);
        }
    }
}


//-------------------------------------------------------------------------------------
void createSLDistortionMesh(SLEye eye, SLGLBuffer& vb, SLGLBuffer& ib)
{
    // fill the variables below with useful data from dk2
    HmdRenderInfo hmdri;
    hmdri.HmdType = HmdType_DK2;
    hmdri.ResolutionInPixels.w = 1920;
    hmdri.ResolutionInPixels.h = 1080;
    hmdri.ScreenSizeInMeters.w = 0.125760004f;
    hmdri.ScreenSizeInMeters.h = 0.0707399994f;
    hmdri.ScreenGapSizeInMeters = 0.0f;
    hmdri.CenterFromTopInMeters = 0.0353f;
    hmdri.LensSeparationInMeters = 0.0635f;
    hmdri.LensDiameterInMeters = 0.0399f;
    hmdri.LensSurfaceToMidplateInMeters = 0.01964f;
    hmdri.EyeCups = EyeCup_DK2A;
    hmdri.Shutter.Type = HmdShutter_RollingRightToLeft;
    hmdri.Shutter.VsyncToNextVsync = 0.013157f;
    hmdri.Shutter.VsyncToFirstScanline = 2.73000005e-005f;
    hmdri.Shutter.FirstScanlineToLastScanline = 0.0131f;
    hmdri.Shutter.PixelSettleTime = 0.0f;
    hmdri.Shutter.PixelPersistence = 0.0023f;

    hmdri.EyeLeft.ReliefInMeters = 0.0109f;
    hmdri.EyeLeft.NoseToPupilInMeters = 0.032f;
    hmdri.EyeLeft.Distortion.Eqn = Distortion_CatmullRom10;
    hmdri.EyeLeft.Distortion.K[0] = 1.00300002f;
    hmdri.EyeLeft.Distortion.K[1] = 1.01999998f;
    hmdri.EyeLeft.Distortion.K[2] = 1.04200006f;
    hmdri.EyeLeft.Distortion.K[3] = 1.06599998f;
    hmdri.EyeLeft.Distortion.K[4] = 1.09399998f;
    hmdri.EyeLeft.Distortion.K[5] = 1.12600005f;
    hmdri.EyeLeft.Distortion.K[6] = 1.16199994f;
    hmdri.EyeLeft.Distortion.K[7] = 1.20299995f;
    hmdri.EyeLeft.Distortion.K[8] = 1.25000000f;
    hmdri.EyeLeft.Distortion.K[9] = 1.30999994f;
    hmdri.EyeLeft.Distortion.K[10] = 1.38000000f;
    hmdri.EyeLeft.Distortion.MaxR = 1.00000000f;
    hmdri.EyeLeft.Distortion.MetersPerTanAngleAtCenter = 0.0359999985f;
    hmdri.EyeLeft.Distortion.ChromaticAberration[0] = -0.0123399980f;
    hmdri.EyeLeft.Distortion.ChromaticAberration[1] = -0.0164999980f;
    hmdri.EyeLeft.Distortion.ChromaticAberration[2] = 0.0205899980f;
    hmdri.EyeLeft.Distortion.ChromaticAberration[3] = 0.0164999980f;
    hmdri.EyeLeft.Distortion.InvK[0] = 1.0f;
    hmdri.EyeLeft.Distortion.InvK[1] = 0.964599669f;
    hmdri.EyeLeft.Distortion.InvK[2] = 0.931152463f;
    hmdri.EyeLeft.Distortion.InvK[3] = 0.898376584f;
    hmdri.EyeLeft.Distortion.InvK[4] = 0.867980957f;
    hmdri.EyeLeft.Distortion.InvK[5] = 0.839782715f;
    hmdri.EyeLeft.Distortion.InvK[6] = 0.813964784f;
    hmdri.EyeLeft.Distortion.InvK[7] = 0.789245605f;
    hmdri.EyeLeft.Distortion.InvK[8] = 0.765808105f;
    hmdri.EyeLeft.Distortion.InvK[9] = 0.745178223f;
    hmdri.EyeLeft.Distortion.InvK[10] = 0.724639833f;
    hmdri.EyeLeft.Distortion.MaxInvR = 1.38000000f;

    hmdri.EyeRight.ReliefInMeters = 0.0109f;
    hmdri.EyeRight.NoseToPupilInMeters = 0.032f;
    hmdri.EyeRight.Distortion.Eqn = Distortion_CatmullRom10;
    hmdri.EyeRight.Distortion.K[0] = 1.00300002f;
    hmdri.EyeRight.Distortion.K[1] = 1.01999998f;
    hmdri.EyeRight.Distortion.K[2] = 1.04200006f;
    hmdri.EyeRight.Distortion.K[3] = 1.06599998f;
    hmdri.EyeRight.Distortion.K[4] = 1.09399998f;
    hmdri.EyeRight.Distortion.K[5] = 1.12600005f;
    hmdri.EyeRight.Distortion.K[6] = 1.16199994f;
    hmdri.EyeRight.Distortion.K[7] = 1.20299995f;
    hmdri.EyeRight.Distortion.K[8] = 1.25000000f;
    hmdri.EyeRight.Distortion.K[9] = 1.30999994f;
    hmdri.EyeRight.Distortion.K[10] = 1.38000000f;
    hmdri.EyeRight.Distortion.MaxR = 1.00000000f;
    hmdri.EyeRight.Distortion.MetersPerTanAngleAtCenter = 0.0359999985f;
    hmdri.EyeRight.Distortion.ChromaticAberration[0] = -0.0123399980f;
    hmdri.EyeRight.Distortion.ChromaticAberration[1] = -0.0164999980f;
    hmdri.EyeRight.Distortion.ChromaticAberration[2] = 0.0205899980f;
    hmdri.EyeRight.Distortion.ChromaticAberration[3] = 0.0164999980f;
    hmdri.EyeRight.Distortion.InvK[0] = 1.0f;
    hmdri.EyeRight.Distortion.InvK[1] = 0.964599669f;
    hmdri.EyeRight.Distortion.InvK[2] = 0.931152463f;
    hmdri.EyeRight.Distortion.InvK[3] = 0.898376584f;
    hmdri.EyeRight.Distortion.InvK[4] = 0.867980957f;
    hmdri.EyeRight.Distortion.InvK[5] = 0.839782715f;
    hmdri.EyeRight.Distortion.InvK[6] = 0.813964784f;
    hmdri.EyeRight.Distortion.InvK[7] = 0.789245605f;
    hmdri.EyeRight.Distortion.InvK[8] = 0.765808105f;
    hmdri.EyeRight.Distortion.InvK[9] = 0.745178223f;
    hmdri.EyeRight.Distortion.InvK[10] = 0.724639833f;
    hmdri.EyeRight.Distortion.MaxInvR = 1.38000000f;

    DistortionRenderDesc distortion;
    distortion.Lens.Eqn = Distortion_CatmullRom10;
    distortion.Lens.K[0] = 1.00300002f;
    distortion.Lens.K[1] = 1.01999998f;
    distortion.Lens.K[2] = 1.04200006f;
    distortion.Lens.K[3] = 1.06599998f;
    distortion.Lens.K[4] = 1.09399998f;
    distortion.Lens.K[5] = 1.12600005f;
    distortion.Lens.K[6] = 1.16199994f;
    distortion.Lens.K[7] = 1.20299995f;
    distortion.Lens.K[8] = 1.25000000f;
    distortion.Lens.K[9] = 1.30999994f;
    distortion.Lens.K[10] = 1.38000000f;
    distortion.Lens.MaxR = 1.00000000f;
    distortion.Lens.MetersPerTanAngleAtCenter = 0.0359999985f;
    distortion.Lens.ChromaticAberration[0] = -0.0123399980f;
    distortion.Lens.ChromaticAberration[1] = -0.0164999980f;
    distortion.Lens.ChromaticAberration[2] = 0.0205899980f;
    distortion.Lens.ChromaticAberration[3] = 0.0164999980f;
    distortion.Lens.InvK[0] = 1.0f;
    distortion.Lens.InvK[1] = 0.964599669f;
    distortion.Lens.InvK[2] = 0.931152463f;
    distortion.Lens.InvK[3] = 0.898376584f;
    distortion.Lens.InvK[4] = 0.867980957f;
    distortion.Lens.InvK[5] = 0.839782715f;
    distortion.Lens.InvK[6] = 0.813964784f;
    distortion.Lens.InvK[7] = 0.789245605f;
    distortion.Lens.InvK[8] = 0.765808105f;
    distortion.Lens.InvK[9] = 0.745178223f;
    distortion.Lens.InvK[10] = 0.724639833f;
    distortion.Lens.MaxInvR = 1.38000000f;

    distortion.LensCenter.x = -0.00986003876f;
    distortion.LensCenter.y = 0.000000000f;
    distortion.TanEyeAngleScale.x = 0.873333395f;
    distortion.TanEyeAngleScale.y = 0.982500017f;
    distortion.PixelsPerTanAngleAtCenter.x = 549.618286f;
    distortion.PixelsPerTanAngleAtCenter.y = 549.618286f;

    ovrFovPort fov;
    fov.DownTan = 1.329f;
    fov.UpTan = 1.329f;
    fov.LeftTan = 1.058f;
    fov.RightTan = 1.092f;

    ovrDistortionVertex* vertexData;
    unsigned short* indexData;
    int triangleCount = 0;
    int vertexCount = 0;
    #ifdef SL_GUI_JAVA
    bool rightEye = (eye == rightEye);
    #else
    bool rightEye = (eye == SLEye::rightEye);
    #endif
    ScaleAndOffset2D      eyeToSourceNDC = CreateNDCScaleAndOffsetFromFov(fov);
    eyeToSourceNDC.Scale.x = 0.929788947f;
    eyeToSourceNDC.Scale.y = 0.752283394f;
    eyeToSourceNDC.Offset.x = -0.0156717598f;
    eyeToSourceNDC.Offset.y = 0.0f;
    if(rightEye) {
        eyeToSourceNDC.Offset.x *= -1;
        distortion.LensCenter.x *= -1;
    }

    createSLDistortionMesh((DistortionMeshVertexData**)&vertexData, (uint16_t**)&indexData, 
                           &vertexCount, &triangleCount,
                           rightEye,
                           hmdri, distortion, eyeToSourceNDC);

    int indexCount = triangleCount * 3;


    // Now parse the vertex data and create a render ready vertex buffer from it
    SLGLOcculusDistortionVertex* pVBVerts = new SLGLOcculusDistortionVertex[vertexCount];

    vector<SLuint> tempIndex;

    SLGLOcculusDistortionVertex* v = pVBVerts;
    ovrDistortionVertex * ov = vertexData;
    for ( unsigned vertNum = 0; vertNum < vertexCount; vertNum++ )
    {
        v->screenPosNDC.x = ov->ScreenPosNDC.x;
        v->screenPosNDC.y = ov->ScreenPosNDC.y;

        v->timeWarpFactor = ov->TimeWarpFactor;
        v->vignetteFactor = ov->VignetteFactor;
            
        v->tanEyeAnglesR.x = ov->TanEyeAnglesR.x;
        v->tanEyeAnglesR.y = ov->TanEyeAnglesR.y;

        v->tanEyeAnglesG.x = ov->TanEyeAnglesG.x;
        v->tanEyeAnglesG.y = ov->TanEyeAnglesG.y;

        v->tanEyeAnglesB.x = ov->TanEyeAnglesB.x;
        v->tanEyeAnglesB.y = ov->TanEyeAnglesB.y;
            
        v++; ov++;
    }

    for (unsigned i = 0; i < indexCount; i++)
        tempIndex.push_back(indexData[i]);

    //@todo the SLGLBuffer isn't made for this kind of interleaved usage
    //       rework it so it is easier to use and more dynamic.
    vb.generate(pVBVerts, vertexCount, 10,
                SL_FLOAT, SL_ARRAY_BUFFER, SL_STATIC_DRAW);
    // somehow passing in meshData.pIndexData doesn't work...
    ib.generate(&tempIndex[0], indexCount, 1,
                SL_UNSIGNED_INT, SL_ELEMENT_ARRAY_BUFFER, SL_STATIC_DRAW);


    delete[] pVBVerts;
     
}

//-------------------------------------------------------------------------------------
#endif

#endif
