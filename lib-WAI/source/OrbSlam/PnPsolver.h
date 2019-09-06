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

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include <WAIFrame.h>
#include <WAIMapPoint.h>

namespace ORB_SLAM2
{

class PnPsolver
{
    public:
    PnPsolver(const WAIFrame& F, const vector<WAIMapPoint*>& vpMapPointMatches);

    ~PnPsolver();

    void SetRansacParameters(double probability   = 0.99,
                             int    minInliers    = 8,
                             int    maxIterations = 300,
                             int    minSet        = 4,
                             float  epsilon       = 0.4,
                             float  th2           = 5.991);

    cv::Mat find(vector<bool>& vbInliers, int& nInliers);

    cv::Mat iterate(int           nIterations,
                    bool&         bNoMore,
                    vector<bool>& vbInliers,
                    int&          nInliers);

    private:
    void CheckInliers();
    bool Refine();

    // Functions from the original EPnP code
    void set_maximum_number_of_correspondences(int n);
    void reset_correspondences();
    void add_correspondence(double X,
                            double Y,
                            double Z,
                            double u,
                            double v);

    double compute_pose(double R[3][3], double T[3]);

    void relative_error(double&      rot_err,
                        double&      transl_err,
                        const double Rtrue[3][3],
                        const double ttrue[3],
                        const double Rest[3][3],
                        const double test[3]);

    void   print_pose(const double R[3][3], const double t[3]);
    double reprojection_error(const double R[3][3], const double t[3]);

    void choose_control_points();
    void compute_barycentric_coordinates();
    void fill_M(cv::Mat* M, int row, const double* alphas, double u, double v);
    void compute_ccs(const double* betas, const double* ut);
    void compute_pcs();

    void solve_for_sign();

    void find_betas_approx_1(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas);
    void find_betas_approx_2(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas);
    void find_betas_approx_3(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas);
    void qr_solve(cv::Mat* A, cv::Mat* b, cv::Mat* X);

    double dot(const double* v1, const double* v2);
    double dist2(const double* p1, const double* p2);

    void   compute_rho(double* rho);
    void   compute_L_6x10(const double* ut, double* l_6x10);
    void   gauss_newton(const cv::Mat* L_6x10, const cv::Mat* Rho, double current_betas[4]);
    void   compute_A_and_b_gauss_newton(const double* l_6x10,
                                        const double* rho,
                                        const double        cb[4],
                                        cv::Mat*      A,
                                        cv::Mat*      b);
    double compute_R_and_t(const double* ut,
                           const double* betas,
                           double        R[3][3],
                           double        t[3]);
    void   estimate_R_and_t(double R[3][3], double t[3]);
    void   copy_R_and_t(const double R_dst[3][3],
                        const double t_dst[3],
                        double       R_src[3][3],
                        double       t_src[3]);
    void   mat_to_quat(const double R[3][3], double q[4]);

    double  _uc, _vc, _fu, _fv;
    double *_pws, *_us, *_alphas, *_pcs;
    int     _maxNumCorrespondences;
    int     _numCorrespondences;
    double  _cws[4][3], _ccs[4][3];
    double  _cws_determinant;

    vector<WAIMapPoint*> _mvpMapPointMatches;
    vector<cv::Point2f>  _mvP2D; //!< 2D Points
    vector<float>        _mvSigma2;
    vector<cv::Point3f>  _mvP3Dw;            //!< 3D Points
    vector<size_t>       _mvKeyPointIndices; //!< Index in Frame

    // Current Estimation
    double       _mRi[3][3];
    double       _mti[3];
    cv::Mat      _mTcwi;
    vector<bool> _mvbInliersi;
    int          _mnInliersi;

    // Current Ransac State
    int          _mnIterations;
    vector<bool> _mvbBestInliers;
    int          _mnBestInliers;
    cv::Mat      _mBestTcw;

    // Refined
    cv::Mat      _mRefinedTcw;
    vector<bool> _mvbRefinedInliers;
    int          _mnRefinedInliers;

    int            _mN;                //!< Number of Correspondences
    vector<size_t> _mvAllIndices;      //!< Indices for random selection [0 .. _numCorresp-1]
    double         _mRansacProb;       //!< RANSAC probability
    int            _mRansacMinInliers; //!< RANSAC min inliers
    int            _mRansacMaxIts;     //!< RANSAC max iterations
    float          _mRansacEpsilon;    //!< RANSAC expected inliers/total ratio
    float          _mRansacTh;         //!< RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
    int            _mRansacMinSet;     //!< RANSAC Minimun Set used at each iteration
    vector<float>  _mvMaxError;        //!< Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
};

} //namespace ORB_SLAM

#endif //PNPSOLVER_H
