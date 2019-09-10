/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#include <iostream>

#include <OrbSlam/PnPsolver.h>

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <DUtils/Random.h>
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{

//-----------------------------------------------------------------------------
PnPsolver::PnPsolver(const WAIFrame&             F,
                     const vector<WAIMapPoint*>& vpMapPointMatches) : _pws(0),
                                                                      _us(0),
                                                                      _alphas(0),
                                                                      _pcs(0),
                                                                      _maxNumCorrespondences(0),
                                                                      _numCorrespondences(0),
                                                                      _mnInliersi(0),
                                                                      _mnIterations(0),
                                                                      _mnBestInliers(0),
                                                                      _mN(0)
{
    _mvpMapPointMatches = vpMapPointMatches;
    _mvP2D.reserve(F.mvpMapPoints.size());
    _mvSigma2.reserve(F.mvpMapPoints.size());
    _mvP3Dw.reserve(F.mvpMapPoints.size());
    _mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    _mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx = 0;
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        WAIMapPoint* pMP = vpMapPointMatches[i];

        if (pMP)
        {
            if (!pMP->isBad())
            {
                const cv::KeyPoint& kp = F.mvKeysUn[i];

                _mvP2D.push_back(kp.pt);
                _mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

                //cv::Mat Pos = pMP->GetWorldPos();
                //_mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));
                auto Pos = pMP->worldPosVec();
                _mvP3Dw.emplace_back(cv::Point3f(Pos.x, Pos.y, Pos.z));

                _mvKeyPointIndices.push_back(i);
                _mvAllIndices.push_back(idx);

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    _fu = F.fx;
    _fv = F.fy;
    _uc = F.cx;
    _vc = F.cy;

    SetRansacParameters();
}
//-----------------------------------------------------------------------------
PnPsolver::~PnPsolver()
{
    delete[] _pws;
    delete[] _us;
    delete[] _alphas;
    delete[] _pcs;
}
//-----------------------------------------------------------------------------
void PnPsolver::SetRansacParameters(double probability,
                                    int    minInliers,
                                    int    maxIterations,
                                    int    minSet,
                                    float  epsilon,
                                    float  th2)
{
    _mRansacProb       = probability;
    _mRansacMinInliers = minInliers;
    _mRansacMaxIts     = maxIterations;
    _mRansacEpsilon    = epsilon;
    _mRansacMinSet     = minSet;

    _mN = (int)_mvP2D.size(); // number of correspondences

    _mvbInliersi.resize(_mN);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = (int)(_mN * _mRansacEpsilon);
    if (nMinInliers < _mRansacMinInliers)
        nMinInliers = _mRansacMinInliers;
    if (nMinInliers < minSet)
        nMinInliers = minSet;
    _mRansacMinInliers = nMinInliers;

    if (_mRansacEpsilon < (float)_mRansacMinInliers / _mN)
        _mRansacEpsilon = (float)_mRansacMinInliers / _mN;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (_mRansacMinInliers == _mN)
        nIterations = 1;
    else
        nIterations = (int)ceil(log(1 - _mRansacProb) / log(1 - pow(_mRansacEpsilon, 3)));

    _mRansacMaxIts = max(1, min(nIterations, _mRansacMaxIts));

    _mvMaxError.resize(_mvSigma2.size());
    for (size_t i = 0; i < _mvSigma2.size(); i++)
        _mvMaxError[i] = _mvSigma2[i] * th2;
}
//-----------------------------------------------------------------------------
cv::Mat PnPsolver::find(vector<bool>& vbInliers, int& nInliers)
{
    bool bFlag;
    return iterate(_mRansacMaxIts, bFlag, vbInliers, nInliers);
}
//-----------------------------------------------------------------------------
cv::Mat PnPsolver::iterate(int           nIterations,
                           bool&         bNoMore,
                           vector<bool>& vbInliers,
                           int&          nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    set_maximum_number_of_correspondences(_mRansacMinSet);

    if (_mN < _mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while (_mnIterations < _mRansacMaxIts || nCurrentIterations < nIterations)
    {
        nCurrentIterations++;
        _mnIterations++;
        reset_correspondences();

        vAvailableIndices = _mvAllIndices;

        // Get min set of points
        for (int i = 0; i < _mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, (int)vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            add_correspondence(_mvP3Dw[idx].x, _mvP3Dw[idx].y, _mvP3Dw[idx].z, _mvP2D[idx].x, _mvP2D[idx].y);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        compute_pose(_mRi, _mti);

        // Check inliers
        CheckInliers();

        if (_mnInliersi >= _mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            if (_mnInliersi > _mnBestInliers)
            {
                _mvbBestInliers = _mvbInliersi;
                _mnBestInliers  = _mnInliersi;

                cv::Mat Rcw(3, 3, CV_64F, _mRi);
                cv::Mat tcw(3, 1, CV_64F, _mti);
                Rcw.convertTo(Rcw, CV_32F);
                tcw.convertTo(tcw, CV_32F);
                _mBestTcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(_mBestTcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(_mBestTcw.rowRange(0, 3).col(3));
            }

            if (Refine())
            {
                nInliers  = _mnRefinedInliers;
                vbInliers = vector<bool>(_mvpMapPointMatches.size(), false);
                for (int i = 0; i < _mN; i++)
                {
                    if (_mvbRefinedInliers[i])
                        vbInliers[_mvKeyPointIndices[i]] = true;
                }
                return _mRefinedTcw.clone();
            }
        }
    }

    if (_mnIterations >= _mRansacMaxIts)
    {
        bNoMore = true;
        if (_mnBestInliers >= _mRansacMinInliers)
        {
            nInliers  = _mnBestInliers;
            vbInliers = vector<bool>(_mvpMapPointMatches.size(), false);
            for (int i = 0; i < _mN; i++)
            {
                if (_mvbBestInliers[i])
                    vbInliers[_mvKeyPointIndices[i]] = true;
            }
            return _mBestTcw.clone();
        }
    }

    return cv::Mat();
}
//-----------------------------------------------------------------------------
bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(_mvbBestInliers.size());

    for (size_t i = 0; i < _mvbBestInliers.size(); i++)
    {
        if (_mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for (int idx : vIndices)
    {
        add_correspondence(_mvP3Dw[idx].x, _mvP3Dw[idx].y, _mvP3Dw[idx].z, _mvP2D[idx].x, _mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(_mRi, _mti);

    // Check inliers
    CheckInliers();

    _mnRefinedInliers  = _mnInliersi;
    _mvbRefinedInliers = _mvbInliersi;

    if (_mnInliersi > _mRansacMinInliers)
    {
        cv::Mat Rcw(3, 3, CV_64F, _mRi);
        cv::Mat tcw(3, 1, CV_64F, _mti);
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        _mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(_mRefinedTcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(_mRefinedTcw.rowRange(0, 3).col(3));
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
void PnPsolver::CheckInliers()
{
    _mnInliersi = 0;

    for (int i = 0; i < _mN; i++)
    {
        cv::Point3f P3Dw = _mvP3Dw[i];
        cv::Point2f P2D  = _mvP2D[i];

        float Xc    = (float)(_mRi[0][0] * P3Dw.x + _mRi[0][1] * P3Dw.y + _mRi[0][2] * P3Dw.z + _mti[0]);
        float Yc    = (float)(_mRi[1][0] * P3Dw.x + _mRi[1][1] * P3Dw.y + _mRi[1][2] * P3Dw.z + _mti[1]);
        float invZc = (float)(1.0f / (_mRi[2][0] * P3Dw.x + _mRi[2][1] * P3Dw.y + _mRi[2][2] * P3Dw.z + _mti[2]));

        double ue = _uc + _fu * Xc * invZc;
        double ve = _vc + _fv * Yc * invZc;

        float distX = (float)(P2D.x - ue);
        float distY = (float)(P2D.y - ve);

        float error2 = distX * distX + distY * distY;

        if (error2 < _mvMaxError[i])
        {
            _mvbInliersi[i] = true;
            _mnInliersi++;
        }
        else
        {
            _mvbInliersi[i] = false;
        }
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
    if (_maxNumCorrespondences < n)
    {
        delete[] _pws;
        delete[] _us;
        delete[] _alphas;
        delete[] _pcs;

        _maxNumCorrespondences = n;
        _pws                   = new double[3 * _maxNumCorrespondences];
        _us                    = new double[2 * _maxNumCorrespondences];
        _alphas                = new double[4 * _maxNumCorrespondences];
        _pcs                   = new double[3 * _maxNumCorrespondences];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::reset_correspondences()
{
    _numCorrespondences = 0;
}
//-----------------------------------------------------------------------------
void PnPsolver::add_correspondence(double X,
                                   double Y,
                                   double Z,
                                   double u,
                                   double v)
{
    _pws[3 * _numCorrespondences]     = X;
    _pws[3 * _numCorrespondences + 1] = Y;
    _pws[3 * _numCorrespondences + 2] = Z;

    _us[2 * _numCorrespondences]     = u;
    _us[2 * _numCorrespondences + 1] = v;

    _numCorrespondences++;
}
//-----------------------------------------------------------------------------
void PnPsolver::choose_control_points()
{
    // Take C0 as the reference points centroid:
    _cws[0][0] = _cws[0][1] = _cws[0][2] = 0;
    for (int i = 0; i < _numCorrespondences; i++)
        for (int j = 0; j < 3; j++)
            _cws[0][j] += _pws[3 * i + j];

    for (int j = 0; j < 3; j++)
        _cws[0][j] /= _numCorrespondences;

    // Take C1, C2, and C3 from PCA on the reference points:
    //#if CV_VERSION_MAJOR < 4
    //    CvMat* PW0 = cvCreateMat(_numCorrespondences, 3, CV_64F);
    //#else
    cv::Mat PW0(_numCorrespondences, 3, CV_64F);
    //#endif

    double  pw0tpw0[3 * 3], dc[3], uct[3 * 3], vct[3 * 3];
    cv::Mat PW0tPW0 = cv::Mat(3, 3, CV_64F, pw0tpw0);
    cv::Mat DC      = cv::Mat(3, 1, CV_64F, dc);
    cv::Mat UCt     = cv::Mat(3, 3, CV_64F, uct);
    cv::Mat VCt     = cv::Mat(3, 3, CV_64F, vct);

    //#if CV_VERSION_MAJOR < 4
    //    for (int i = 0; i < _numCorrespondences; i++)
    //        for (int j = 0; j < 3; j++)
    //            PW0->data.db[3 * i + j] = _pws[3 * i + j] - _cws[0][j];
    //#else
    double* db = PW0.ptr<double>(0);
    for (int i = 0; i < _numCorrespondences; i++)
        for (int j = 0; j < 3; j++)
            db[3 * i + j] = _pws[3 * i + j] - _cws[0][j];
            //#endif

#if CV_VERSION_MAJOR < 4
    cvMulTransposed(&PW0, &PW0tPW0, 1);
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    //cvReleaseMat(&PW0);
#else
    cv::mulTransposed(PW0, PW0tPW0, true);
    cv::SVD::compute(PW0tPW0, DC, UCt, VCt, cv::SVD::MODIFY_A /*| cv::SVD::FULL_UV*/);
    cv::transpose(UCt, UCt);
#endif

    for (int i = 1; i < 4; i++)
    {
        double k = sqrt(dc[i - 1] / _numCorrespondences);
        for (int j = 0; j < 3; j++)
            _cws[i][j] = _cws[0][j] + k * uct[3 * (i - 1) + j];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_barycentric_coordinates()
{
    double  cc[3 * 3], cc_inv[3 * 3];
    cv::Mat CC     = cv::Mat(3, 3, CV_64F, cc);
    cv::Mat CC_inv = cv::Mat(3, 3, CV_64F, cc_inv);

    for (int i = 0; i < 3; i++)
        for (int j = 1; j < 4; j++)
            cc[3 * i + j - 1] = _cws[j][i] - _cws[0][i];

    //TODO: delete old code
    //cvInvert(&CC, &CC_inv, CV_SVD);
    cv::invert(CC, CC_inv, cv::DECOMP_SVD);

    double* ci = cc_inv;
    for (int i = 0; i < _numCorrespondences; i++)
    {
        double* pi = _pws + 3 * i;
        double* a  = _alphas + 4 * i;

        for (int j = 0; j < 3; j++)
            a[1 + j] =
              ci[3 * j] * (pi[0] - _cws[0][0]) +
              ci[3 * j + 1] * (pi[1] - _cws[0][1]) +
              ci[3 * j + 2] * (pi[2] - _cws[0][2]);

        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::fill_M(cv::Mat*      M,
                       const int     row,
                       const double* as,
                       const double  u,
                       const double  v)
{
    //TODO: delete old code
    //double* M1 = M->data.db + row * 12;
    double* M1 = M->ptr<double>(0) + row * 12;
    double* M2 = M1 + 12;

    for (int i = 0; i < 4; i++)
    {
        M1[3 * i]     = as[i] * _fu;
        M1[3 * i + 1] = 0.0;
        M1[3 * i + 2] = as[i] * (_uc - u);

        M2[3 * i]     = 0.0;
        M2[3 * i + 1] = as[i] * _fv;
        M2[3 * i + 2] = as[i] * (_vc - v);
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_ccs(const double* betas, const double* ut)
{
    for (auto& _cc : _ccs)
        _cc[0] = _cc[1] = _cc[2] = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        const double* v = ut + 12 * (11 - i);
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                _ccs[j][k] += betas[i] * v[3 * j + k];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_pcs()
{
    for (int i = 0; i < _numCorrespondences; i++)
    {
        double* a  = _alphas + 4 * i;
        double* pc = _pcs + 3 * i;

        for (int j = 0; j < 3; j++)
            pc[j] = a[0] * _ccs[0][j] + a[1] * _ccs[1][j] + a[2] * _ccs[2][j] + a[3] * _ccs[3][j];
    }
}
//-----------------------------------------------------------------------------
double PnPsolver::compute_pose(double R[3][3], double t[3])
{
    choose_control_points();
    compute_barycentric_coordinates();

    //TODO: delete old code
    //CvMat* M = cvCreateMat(2 * _numCorrespondences, 12, CV_64F);
    cv::Mat M(2 * _numCorrespondences, 12, CV_64F);

    for (int i = 0; i < _numCorrespondences; i++)
        fill_M(&M, 2 * i, _alphas + 4 * i, _us[2 * i], _us[2 * i + 1]);

    double  mtm[12 * 12], d[12], ut[12 * 12], vt[12 * 12];
    cv::Mat MtM = cv::Mat(12, 12, CV_64F, mtm);
    cv::Mat D   = cv::Mat(12, 1, CV_64F, d);
    cv::Mat Ut  = cv::Mat(12, 12, CV_64F, ut);
    cv::Mat Vt  = cv::Mat(12, 12, CV_64F, vt);

#if CV_VERSION_MAJOR < 4
    cvMulTransposed(&M, &MtM, 1);
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    //cvReleaseMat(&M);
#else
    cv::mulTransposed(M, MtM, true);
    cv::SVD::compute(MtM, D, Ut, Vt, cv::SVD::MODIFY_A /*| cv::SVD::FULL_UV*/);
    cv::transpose(Ut, Ut);
#endif

    double  l_6x10[6 * 10], rho[6];
    cv::Mat L_6x10 = cv::Mat(6, 10, CV_64F, l_6x10);
    cv::Mat Rho    = cv::Mat(6, 1, CV_64F, rho);

    compute_L_6x10(ut, l_6x10);
    compute_rho(rho);

    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];

    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    copy_R_and_t(Rs[N], ts[N], R, t);

    return rep_errors[N];
}
//-----------------------------------------------------------------------------
void PnPsolver::copy_R_and_t(const double R_src[3][3],
                             const double t_src[3],
                             double       R_dst[3][3],
                             double       t_dst[3])
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            R_dst[i][j] = R_src[i][j];
        t_dst[i] = t_src[i];
    }
}
//-----------------------------------------------------------------------------
double PnPsolver::dist2(const double* p1, const double* p2)
{
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
           (p1[1] - p2[1]) * (p1[1] - p2[1]) +
           (p1[2] - p2[2]) * (p1[2] - p2[2]);
}
//-----------------------------------------------------------------------------
double PnPsolver::dot(const double* v1, const double* v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
//-----------------------------------------------------------------------------
double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
    double sum2 = 0.0;

    for (int i = 0; i < _numCorrespondences; i++)
    {
        double* pw     = _pws + 3 * i;
        double  Xc     = dot(R[0], pw) + t[0];
        double  Yc     = dot(R[1], pw) + t[1];
        double  inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
        double  ue     = _uc + _fu * Xc * inv_Zc;
        double  ve     = _vc + _fv * Yc * inv_Zc;
        double  u = _us[2 * i], v = _us[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }

    return sum2 / _numCorrespondences;
}
//-----------------------------------------------------------------------------
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
    double pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for (int i = 0; i < _numCorrespondences; i++)
    {
        const double* pc = _pcs + 3 * i;
        const double* pw = _pws + 3 * i;

        for (int j = 0; j < 3; j++)
        {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }
    for (int j = 0; j < 3; j++)
    {
        pc0[j] /= _numCorrespondences;
        pw0[j] /= _numCorrespondences;
    }

    double  abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
    cv::Mat ABt   = cv::Mat(3, 3, CV_64F, abt);
    cv::Mat ABt_D = cv::Mat(3, 1, CV_64F, abt_d);
    cv::Mat ABt_U = cv::Mat(3, 3, CV_64F, abt_u);
    cv::Mat ABt_V = cv::Mat(3, 3, CV_64F, abt_v);

#if CV_VERSION_MAJOR < 4
    cvSetZero(&ABt);
#else
    ABt.setTo(0);
#endif

    for (int i = 0; i < _numCorrespondences; i++)
    {
        double* pc = _pcs + 3 * i;
        double* pw = _pws + 3 * i;

        for (int j = 0; j < 3; j++)
        {
            abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

#if CV_VERSION_MAJOR < 4
    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);
#else
    cv::SVD::compute(ABt, ABt_D, ABt_U, ABt_V, cv::SVD::MODIFY_A);
#endif

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

    const double det =
      R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
      R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    if (det < 0)
    {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    t[0] = pc0[0] - dot(R[0], pw0);
    t[1] = pc0[1] - dot(R[1], pw0);
    t[2] = pc0[2] - dot(R[2], pw0);
}
//-----------------------------------------------------------------------------
void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
    cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
    cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
    cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}
//-----------------------------------------------------------------------------
void PnPsolver::solve_for_sign()
{
    if (_pcs[2] < 0.0)
    {
        for (auto& _cc : _ccs)
            for (double& j : _cc)
                j = -j;

        for (int i = 0; i < _numCorrespondences; i++)
        {
            _pcs[3 * i]     = -_pcs[3 * i];
            _pcs[3 * i + 1] = -_pcs[3 * i + 1];
            _pcs[3 * i + 2] = -_pcs[3 * i + 2];
        }
    }
}
//-----------------------------------------------------------------------------
double PnPsolver::compute_R_and_t(const double* ut, const double* betas, double R[3][3], double t[3])
{
    compute_ccs(betas, ut);
    compute_pcs();

    solve_for_sign();

    estimate_R_and_t(R, t);

    return reprojection_error(R, t);
}
//-----------------------------------------------------------------------------
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
void PnPsolver::find_betas_approx_1(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double  l_6x4[6 * 4], b4[4];
    cv::Mat L_6x4 = cv::Mat(6, 4, CV_64F, l_6x4);
    cv::Mat B4    = cv::Mat(4, 1, CV_64F, b4);

    for (int i = 0; i < 6; i++)
    {
        //TODO: delete old code
        //cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
        //cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
        //cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
        //cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));

        L_6x4.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x4.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x4.at<double>(i, 2) = L_6x10->at<double>(i, 3);
        L_6x4.at<double>(i, 3) = L_6x10->at<double>(i, 6);
    }

    //TODO: Check correctness
    //cvSolve(&L_6x4, Rho, &B4, CV_SVD);
    cv::solve(L_6x4, *Rho, B4, cv::DECOMP_SVD);

    if (b4[0] < 0)
    {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    }
    else
    {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}
//-----------------------------------------------------------------------------
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
void PnPsolver::find_betas_approx_2(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double  l_6x3[6 * 3], b3[3];
    cv::Mat L_6x3 = cv::Mat(6, 3, CV_64F, l_6x3);
    cv::Mat B3    = cv::Mat(3, 1, CV_64F, b3);

    for (int i = 0; i < 6; i++)
    {
        //TODO: delete old code
        //cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
        //cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
        //cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));

        L_6x3.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x3.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x3.at<double>(i, 2) = L_6x10->at<double>(i, 3);
    }

    //TODO: Check correctness
    //cvSolve(&L_6x3, Rho, &B3, CV_SVD);
    cv::solve(L_6x3, *Rho, B3, cv::DECOMP_SVD);

    if (b3[0] < 0)
    {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    }
    else
    {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}
//-----------------------------------------------------------------------------
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
void PnPsolver::find_betas_approx_3(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double  l_6x5[6 * 5], b5[5];
    cv::Mat L_6x5 = cv::Mat(6, 5, CV_64F, l_6x5);
    cv::Mat B5    = cv::Mat(5, 1, CV_64F, b5);

    for (int i = 0; i < 6; i++)
    {
        //TODO: delete old code
        //cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
        //cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
        //cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
        //cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
        //cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));

        L_6x5.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x5.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x5.at<double>(i, 2) = L_6x10->at<double>(i, 2);
        L_6x5.at<double>(i, 3) = L_6x10->at<double>(i, 3);
        L_6x5.at<double>(i, 4) = L_6x10->at<double>(i, 4);
    }

    //TODO: Check correctness
    //cvSolve(&L_6x5, Rho, &B5, CV_SVD);
    cv::solve(L_6x5, *Rho, B5, cv::DECOMP_SVD);

    if (b5[0] < 0)
    {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else
    {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_L_6x10(const double* ut, double* l_6x10)
{
    const double* v[4];

    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    double dv[4][6][3];

    for (int i = 0; i < 4; i++)
    {
        int a = 0, b = 1;
        for (int j = 0; j < 6; j++)
        {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

            b++;
            if (b > 3)
            {
                a++;
                b = a + 1;
            }
        }
    }

    for (int i = 0; i < 6; i++)
    {
        double* row = l_6x10 + 10 * i;

        row[0] = dot(dv[0][i], dv[0][i]);
        row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
        row[2] = dot(dv[1][i], dv[1][i]);
        row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
        row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
        row[5] = dot(dv[2][i], dv[2][i]);
        row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
        row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
        row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
        row[9] = dot(dv[3][i], dv[3][i]);
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_rho(double* rho)
{
    rho[0] = dist2(_cws[0], _cws[1]);
    rho[1] = dist2(_cws[0], _cws[2]);
    rho[2] = dist2(_cws[0], _cws[3]);
    rho[3] = dist2(_cws[1], _cws[2]);
    rho[4] = dist2(_cws[1], _cws[3]);
    rho[5] = dist2(_cws[2], _cws[3]);
}
//-----------------------------------------------------------------------------
void PnPsolver::compute_A_and_b_gauss_newton(const double* l_6x10,
                                             const double* rho,
                                             const double  betas[4],
                                             cv::Mat*      A,
                                             cv::Mat*      b)
{
    double* db = A->ptr<double>(0);
    for (int i = 0; i < 6; i++)
    {
        const double* rowL = l_6x10 + i * 10;

        //TODO: delete old code
        //double* rowA = A->data.db + i * 4;
        double* rowA = db + i * 4;

        rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        //cvmSet(b, i, 0, rho[i] - (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] + rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2] + rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] + rowL[6] * betas[0] * betas[3] + rowL[7] * betas[1] * betas[3] + rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]));
        b->at<double>(i, 0) = rho[i] -
                              (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] +
                               rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2] +
                               rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] +
                               rowL[6] * betas[0] * betas[3] + rowL[7] * betas[1] * betas[3] +
                               rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]);
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::gauss_newton(const cv::Mat* L_6x10,
                             const cv::Mat* Rho,
                             double         betas[4])
{
    const int iterations_number = 5;

    double  a[6 * 4], b[6], x[4];
    cv::Mat A = cv::Mat(6, 4, CV_64F, a);
    cv::Mat B = cv::Mat(6, 1, CV_64F, b);
    cv::Mat X = cv::Mat(4, 1, CV_64F, x);

    for (int k = 0; k < iterations_number; k++)
    {
        //TODO: delete old code
        //compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);
        compute_A_and_b_gauss_newton(L_6x10->ptr<double>(0), Rho->ptr<double>(0), betas, &A, &B);

        qr_solve(&A, &B, &X);

        for (int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::qr_solve(cv::Mat* A, cv::Mat* b, cv::Mat* X)
{
    static int     max_nr = 0;
    static double *A1, *A2;

    const int nr = A->rows;
    const int nc = A->cols;

    if (max_nr != 0 && max_nr < nr)
    {
        delete[] A1;
        delete[] A2;
    }
    if (max_nr < nr)
    {
        max_nr = nr;
        A1     = new double[nr];
        A2     = new double[nr];
    }

    //TODO: delete old code
    //double* pA = A->data.db;
    double* pA = A->ptr<double>(0);

    double* ppAkk = pA;
    for (int k = 0; k < nc; k++)
    {
        double *ppAik = ppAkk, eta = fabs(*ppAik);
        for (int i = k + 1; i < nr; i++)
        {
            double elt = fabs(*ppAik);
            if (eta < elt) eta = elt;
            ppAik += nc;
        }

        if (eta == 0)
        {
            A1[k] = A2[k] = 0.0;
            cerr << "God damnit, A is singular, this shouldn't happen." << endl;
            return;
        }
        else
        {
            double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
            for (int i = k; i < nr; i++)
            {
                *ppAik *= inv_eta;
                sum += *ppAik * *ppAik;
                ppAik += nc;
            }
            double sigma = sqrt(sum);
            if (*ppAkk < 0)
                sigma = -sigma;
            *ppAkk += sigma;
            A1[k] = sigma * *ppAkk;
            A2[k] = -eta * sigma;
            for (int j = k + 1; j < nc; j++)
            {
                double *ppAik = ppAkk, sum = 0;
                for (int i = k; i < nr; i++)
                {
                    sum += *ppAik * ppAik[j - k];
                    ppAik += nc;
                }
                double tau = sum / A1[k];
                ppAik      = ppAkk;
                for (int i = k; i < nr; i++)
                {
                    ppAik[j - k] -= tau * *ppAik;
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;
    }

    // b <- Qt b
    double* ppAjj = pA;

    //TODO: delete old code
    //double *pb = b->data.db;
    double* pb = b->ptr<double>(0);

    for (int j = 0; j < nc; j++)
    {
        double *ppAij = ppAjj, tau = 0;
        for (int i = j; i < nr; i++)
        {
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1[j];
        ppAij = ppAjj;
        for (int i = j; i < nr; i++)
        {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b

    //TODO: delete old code
    //double* pX = X->data.db;
    double* pX = X->ptr<double>(0);

    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for (int i = nc - 2; i >= 0; i--)
    {
        double *ppAij = pA + i * nc + (i + 1), sum = 0;

        for (int j = i + 1; j < nc; j++)
        {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2[i];
    }
}
//-----------------------------------------------------------------------------
void PnPsolver::relative_error(double&      rot_err,
                               double&      transl_err,
                               const double Rtrue[3][3],
                               const double ttrue[3],
                               const double Rest[3][3],
                               const double test[3])
{
    double qtrue[4], qest[4];

    mat_to_quat(Rtrue, qtrue);
    mat_to_quat(Rest, qest);

    double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
                           (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
                           (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
                           (qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
                      sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
                           (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
                           (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
                           (qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
                      sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    rot_err = min(rot_err1, rot_err2);

    transl_err =
      sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
           (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
           (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
      sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}
//-----------------------------------------------------------------------------
void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
    double tr = R[0][0] + R[1][1] + R[2][2];
    double n4;

    if (tr > 0.0f)
    {
        q[0] = R[1][2] - R[2][1];
        q[1] = R[2][0] - R[0][2];
        q[2] = R[0][1] - R[1][0];
        q[3] = tr + 1.0f;
        n4   = q[3];
    }
    else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2]))
    {
        q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
        q[1] = R[1][0] + R[0][1];
        q[2] = R[2][0] + R[0][2];
        q[3] = R[1][2] - R[2][1];
        n4   = q[0];
    }
    else if (R[1][1] > R[2][2])
    {
        q[0] = R[1][0] + R[0][1];
        q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
        q[2] = R[2][1] + R[1][2];
        q[3] = R[2][0] - R[0][2];
        n4   = q[1];
    }
    else
    {
        q[0] = R[2][0] + R[0][2];
        q[1] = R[2][1] + R[1][2];
        q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
        q[3] = R[0][1] - R[1][0];
        n4   = q[2];
    }
    double scale = 0.5f / double(sqrt(n4));

    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}
//-----------------------------------------------------------------------------
} //namespace ORB_SLAM
