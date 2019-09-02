#include "WAIPnPSolver.h"

/**
* This file is modified part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*/

static PnPSolver initializePnPSolver(const std::vector<MapPoint*>&    frameMapPoints,
                                     const std::vector<cv::KeyPoint>& frameUndistortedKeyPoints,
                                     const std::vector<r32>&          sigmaSquared,
                                     const std::vector<MapPoint*>&    vpMapPointMatches,
                                     r64                              probability,
                                     i32                              minInliers,
                                     i32                              maxIterations,
                                     i32                              minSet,
                                     r32                              epsilon,
                                     r32                              th2,
                                     r32                              fx,
                                     r32                              fy,
                                     r32                              cx,
                                     r32                              cy)
{
    PnPSolver result = {};

    result.mvpMapPointMatches = vpMapPointMatches;
    result.mvP2D.reserve(frameMapPoints.size());
    result.mvSigma2.reserve(frameMapPoints.size());
    result.mvP3Dw.reserve(frameMapPoints.size());
    result.mvKeyPointIndices.reserve(frameMapPoints.size());
    result.mvAllIndices.reserve(frameMapPoints.size());

    int idx = 0;
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if (pMP)
        {
            if (!pMP->bad)
            {
                const cv::KeyPoint& kp = frameUndistortedKeyPoints[i];

                result.mvP2D.push_back(kp.pt);
                result.mvSigma2.push_back(sigmaSquared[kp.octave]);

                cv::Mat Pos = pMP->position;
                result.mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

                result.mvKeyPointIndices.push_back(i);
                result.mvAllIndices.push_back(idx);

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    result.fu = fx;
    result.fv = fy;
    result.uc = cx;
    result.vc = cy;

    result.ransacParameters.probability   = probability;
    result.ransacParameters.minInliers    = minInliers;
    result.ransacParameters.maxIterations = maxIterations;
    result.ransacParameters.minSet        = minSet;
    result.ransacParameters.epsilon       = epsilon;
    result.ransacParameters.th2           = th2;

    result.pointCount = (int)result.mvP2D.size(); // number of correspondences

    result.mvbInliersi.resize(result.pointCount);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = result.pointCount * epsilon;
    if (nMinInliers < minInliers)
    {
        nMinInliers = minInliers;
    }
    if (nMinInliers < minSet)
    {
        nMinInliers = minSet;
    }
    result.ransacParameters.minInliers = nMinInliers;

    if (epsilon < (float)nMinInliers / result.pointCount)
        epsilon = (float)nMinInliers / result.pointCount;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (nMinInliers == result.pointCount)
    {
        nIterations = 1;
    }
    else
    {
        nIterations = (int)ceil(log(1 - probability) / log(1 - pow(epsilon, 3)));
    }

    result.ransacParameters.maxIterations = std::max(1, std::min(nIterations, maxIterations));

    result.mvMaxError.resize(result.mvSigma2.size());
    for (size_t i = 0; i < result.mvSigma2.size(); i++)
    {
        result.mvMaxError[i] = result.mvSigma2[i] * th2;
    }

    return result;
}

static void chooseControlPointsForPnPSolver(PnPSolver* solver)
{
    // Take C0 as the reference points centroid:
    solver->cws[0][0] = solver->cws[0][1] = solver->cws[0][2] = 0;
    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            solver->cws[0][j] += solver->pws[3 * i + j];
        }
    }

    for (int j = 0; j < 3; j++)
    {
        solver->cws[0][j] /= solver->numberOfCorrespondences;
    }

    // Take C1, C2, and C3 from PCA on the reference points:
    CvMat* PW0 = cvCreateMat(solver->numberOfCorrespondences, 3, CV_64F);

    double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
    CvMat  PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
    CvMat  DC      = cvMat(3, 1, CV_64F, dc);
    CvMat  UCt     = cvMat(3, 3, CV_64F, uct);

    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            PW0->data.db[3 * i + j] = solver->pws[3 * i + j] - solver->cws[0][j];
        }
    }

    cvMulTransposed(PW0, &PW0tPW0, 1);
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

    cvReleaseMat(&PW0);

    for (int i = 1; i < 4; i++)
    {
        double k = sqrt(dc[i - 1] / solver->numberOfCorrespondences);
        for (int j = 0; j < 3; j++)
        {
            solver->cws[i][j] = solver->cws[0][j] + k * uct[3 * (i - 1) + j];
        }
    }
}

static void setMaximumNumberOfPnPSolverCorrespondences(PnPSolver* solver,
                                                       i32        n)
{
    if (solver->maximumNumberOfCorrespondences < n)
    {
        if (solver->pws != 0) delete[] solver->pws;
        if (solver->us != 0) delete[] solver->us;
        if (solver->alphas != 0) delete[] solver->alphas;
        if (solver->pcs != 0) delete[] solver->pcs;

        solver->maximumNumberOfCorrespondences = n;
        solver->pws                            = new r64[3 * solver->maximumNumberOfCorrespondences];
        solver->us                             = new r64[2 * solver->maximumNumberOfCorrespondences];
        solver->alphas                         = new r64[4 * solver->maximumNumberOfCorrespondences];
        solver->pcs                            = new r64[3 * solver->maximumNumberOfCorrespondences];
    }
}

static void computeBarycentriyCoordinatesForPnPSolver(PnPSolver* solver)
{
    double cc[3 * 3], cc_inv[3 * 3];
    CvMat  CC     = cvMat(3, 3, CV_64F, cc);
    CvMat  CC_inv = cvMat(3, 3, CV_64F, cc_inv);

    for (int i = 0; i < 3; i++)
        for (int j = 1; j < 4; j++)
            cc[3 * i + j - 1] = solver->cws[j][i] - solver->cws[0][i];

    cvInvert(&CC, &CC_inv, CV_SVD);
    double* ci = cc_inv;
    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        double* pi = solver->pws + 3 * i;
        double* a  = solver->alphas + 4 * i;

        for (int j = 0; j < 3; j++)
            a[1 + j] =
              ci[3 * j] * (pi[0] - solver->cws[0][0]) +
              ci[3 * j + 1] * (pi[1] - solver->cws[0][1]) +
              ci[3 * j + 2] * (pi[2] - solver->cws[0][2]);
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

static r64 pnpSolverDistSquared(const r64* p1,
                                const r64* p2)
{
    r64 result = (p1[0] - p2[0]) * (p1[0] - p2[0]) +
                 (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                 (p1[2] - p2[2]) * (p1[2] - p2[2]);

    return result;
}

static r64 pnpSolverDot(const r64* v1,
                        const r64* v2)
{
    r64 result = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

    return result;
}

static void pnpSolverQrSolve(CvMat* A, CvMat* b, CvMat* X)
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

    double *pA = A->data.db, *ppAkk = pA;
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
            WAI_LOG("God damnit, A is singular, this shouldn't happen.");
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
    double *ppAjj = pA, *pb = b->data.db;
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
    double* pX = X->data.db;
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

static void pnpSolverGaussNewton(const CvMat* L_6x10,
                                 const CvMat* Rho,
                                 double       betas[4])
{
    const int iterations_number = 5;

    double a[6 * 4], b[6], x[4];
    CvMat  A = cvMat(6, 4, CV_64F, a);
    CvMat  B = cvMat(6, 1, CV_64F, b);
    CvMat  X = cvMat(4, 1, CV_64F, x);

    for (int k = 0; k < iterations_number; k++)
    {
        //compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);
        for (int i = 0; i < 6; i++)
        {
            const double* rowL = L_6x10->data.db + i * 10;
            double*       rowA = A.data.db + i * 4;

            rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
            rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
            rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
            rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

            cvmSet(&B, i, 0, Rho->data.db[i] - (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] + rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2] + rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] + rowL[6] * betas[0] * betas[3] + rowL[7] * betas[1] * betas[3] + rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]));
        }

        pnpSolverQrSolve(&A, &B, &X);

        for (int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}

static void pnpSolverEstimateRAndT(const PnPSolver* solver, double R[3][3], double t[3])
{
    double pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        const double* pc = solver->pcs + 3 * i;
        const double* pw = solver->pws + 3 * i;

        for (int j = 0; j < 3; j++)
        {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }
    for (int j = 0; j < 3; j++)
    {
        pc0[j] /= solver->numberOfCorrespondences;
        pw0[j] /= solver->numberOfCorrespondences;
    }

    double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
    CvMat  ABt   = cvMat(3, 3, CV_64F, abt);
    CvMat  ABt_D = cvMat(3, 1, CV_64F, abt_d);
    CvMat  ABt_U = cvMat(3, 3, CV_64F, abt_u);
    CvMat  ABt_V = cvMat(3, 3, CV_64F, abt_v);

    cvSetZero(&ABt);
    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        double* pc = solver->pcs + 3 * i;
        double* pw = solver->pws + 3 * i;

        for (int j = 0; j < 3; j++)
        {
            abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = pnpSolverDot(abt_u + 3 * i, abt_v + 3 * j);

    const double det =
      R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
      R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    if (det < 0)
    {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    t[0] = pc0[0] - pnpSolverDot(R[0], pw0);
    t[1] = pc0[1] - pnpSolverDot(R[1], pw0);
    t[2] = pc0[2] - pnpSolverDot(R[2], pw0);
}

static r64 pnpSolverComputeRAndT(PnPSolver*    solver,
                                 const double* ut,
                                 const double* betas,
                                 double        R[3][3],
                                 double        t[3])
{
    for (int i = 0; i < 4; i++)
        solver->ccs[i][0] = solver->ccs[i][1] = solver->ccs[i][2] = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        const double* v = ut + 12 * (11 - i);
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                solver->ccs[j][k] += betas[i] * v[3 * j + k];
    }

    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        double* a  = solver->alphas + 4 * i;
        double* pc = solver->pcs + 3 * i;

        for (int j = 0; j < 3; j++)
            pc[j] = a[0] * solver->ccs[0][j] +
                    a[1] * solver->ccs[1][j] +
                    a[2] * solver->ccs[2][j] +
                    a[3] * solver->ccs[3][j];
    }

    if (solver->pcs[2] < 0.0)
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                solver->ccs[i][j] = -solver->ccs[i][j];

        for (int i = 0; i < solver->numberOfCorrespondences; i++)
        {
            solver->pcs[3 * i]     = -solver->pcs[3 * i];
            solver->pcs[3 * i + 1] = -solver->pcs[3 * i + 1];
            solver->pcs[3 * i + 2] = -solver->pcs[3 * i + 2];
        }
    }

    pnpSolverEstimateRAndT(solver, R, t);

    double sum2 = 0.0;

    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        double* pw     = solver->pws + 3 * i;
        double  Xc     = pnpSolverDot(R[0], pw) + t[0];
        double  Yc     = pnpSolverDot(R[1], pw) + t[1];
        double  inv_Zc = 1.0 / (pnpSolverDot(R[2], pw) + t[2]);
        double  ue     = solver->uc + solver->fu * Xc * inv_Zc;
        double  ve     = solver->vc + solver->fv * Yc * inv_Zc;
        double  u = solver->us[2 * i], v = solver->us[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }

    return sum2 / solver->numberOfCorrespondences;
}

static r64 computePnPSolverPose(PnPSolver* solver,
                                double     R[3][3],
                                double     t[3])
{
    chooseControlPointsForPnPSolver(solver);
    computeBarycentriyCoordinatesForPnPSolver(solver);

    CvMat* M = cvCreateMat(2 * solver->numberOfCorrespondences, 12, CV_64F);

    for (int i = 0; i < solver->numberOfCorrespondences; i++)
    {
        const int     row = 2 * i;
        const double* as  = solver->alphas + 4 * i;
        const double  u   = solver->us[2 * i];
        const double  v   = solver->us[2 * i + 1];

        double* M1 = M->data.db + row * 12;
        double* M2 = M1 + 12;

        for (int i = 0; i < 4; i++)
        {
            M1[3 * i]     = as[i] * solver->fu;
            M1[3 * i + 1] = 0.0;
            M1[3 * i + 2] = as[i] * (solver->uc - u);

            M2[3 * i]     = 0.0;
            M2[3 * i + 1] = as[i] * solver->fv;
            M2[3 * i + 2] = as[i] * (solver->vc - v);
        }
    }

    double mtm[12 * 12], d[12], ut[12 * 12];
    CvMat  MtM = cvMat(12, 12, CV_64F, mtm);
    CvMat  D   = cvMat(12, 1, CV_64F, d);
    CvMat  Ut  = cvMat(12, 12, CV_64F, ut);

    cvMulTransposed(M, &MtM, 1);
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    cvReleaseMat(&M);

    double l_6x10[6 * 10], rho[6];
    CvMat  L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    CvMat  Rho    = cvMat(6, 1, CV_64F, rho);

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

            row[0] = pnpSolverDot(dv[0][i], dv[0][i]);
            row[1] = 2.0f * pnpSolverDot(dv[0][i], dv[1][i]);
            row[2] = pnpSolverDot(dv[1][i], dv[1][i]);
            row[3] = 2.0f * pnpSolverDot(dv[0][i], dv[2][i]);
            row[4] = 2.0f * pnpSolverDot(dv[1][i], dv[2][i]);
            row[5] = pnpSolverDot(dv[2][i], dv[2][i]);
            row[6] = 2.0f * pnpSolverDot(dv[0][i], dv[3][i]);
            row[7] = 2.0f * pnpSolverDot(dv[1][i], dv[3][i]);
            row[8] = 2.0f * pnpSolverDot(dv[2][i], dv[3][i]);
            row[9] = pnpSolverDot(dv[3][i], dv[3][i]);
        }
    }

    rho[0] = pnpSolverDistSquared(solver->cws[0], solver->cws[1]);
    rho[1] = pnpSolverDistSquared(solver->cws[0], solver->cws[2]);
    rho[2] = pnpSolverDistSquared(solver->cws[0], solver->cws[3]);
    rho[3] = pnpSolverDistSquared(solver->cws[1], solver->cws[2]);
    rho[4] = pnpSolverDistSquared(solver->cws[1], solver->cws[3]);
    rho[5] = pnpSolverDistSquared(solver->cws[2], solver->cws[3]);

    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];

    {
        double* betas = Betas[1];

        double l_6x4[6 * 4], b4[4];
        CvMat  L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
        CvMat  B4    = cvMat(4, 1, CV_64F, b4);

        for (int i = 0; i < 6; i++)
        {
            cvmSet(&L_6x4, i, 0, cvmGet(&L_6x10, i, 0));
            cvmSet(&L_6x4, i, 1, cvmGet(&L_6x10, i, 1));
            cvmSet(&L_6x4, i, 2, cvmGet(&L_6x10, i, 3));
            cvmSet(&L_6x4, i, 3, cvmGet(&L_6x10, i, 6));
        }

        cvSolve(&L_6x4, &Rho, &B4, CV_SVD);

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

    pnpSolverGaussNewton(&L_6x10, &Rho, Betas[1]);
    rep_errors[1] = pnpSolverComputeRAndT(solver, ut, Betas[1], Rs[1], ts[1]);

    {
        double* betas = Betas[2];
        double  l_6x3[6 * 3], b3[3];
        CvMat   L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
        CvMat   B3    = cvMat(3, 1, CV_64F, b3);

        for (int i = 0; i < 6; i++)
        {
            cvmSet(&L_6x3, i, 0, cvmGet(&L_6x10, i, 0));
            cvmSet(&L_6x3, i, 1, cvmGet(&L_6x10, i, 1));
            cvmSet(&L_6x3, i, 2, cvmGet(&L_6x10, i, 2));
        }

        cvSolve(&L_6x3, &Rho, &B3, CV_SVD);

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
    pnpSolverGaussNewton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = pnpSolverComputeRAndT(solver, ut, Betas[2], Rs[2], ts[2]);

    {
        double* betas = Betas[3];
        double  l_6x5[6 * 5], b5[5];
        CvMat   L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
        CvMat   B5    = cvMat(5, 1, CV_64F, b5);

        for (int i = 0; i < 6; i++)
        {
            cvmSet(&L_6x5, i, 0, cvmGet(&L_6x10, i, 0));
            cvmSet(&L_6x5, i, 1, cvmGet(&L_6x10, i, 1));
            cvmSet(&L_6x5, i, 2, cvmGet(&L_6x10, i, 2));
            cvmSet(&L_6x5, i, 3, cvmGet(&L_6x10, i, 3));
            cvmSet(&L_6x5, i, 4, cvmGet(&L_6x10, i, 4));
        }

        cvSolve(&L_6x5, &Rho, &B5, CV_SVD);

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
    pnpSolverGaussNewton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = pnpSolverComputeRAndT(solver, ut, Betas[3], Rs[3], ts[3]);

    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            R[i][j] = Rs[N][i][j];
        t[i] = ts[N][i];
    }

    return rep_errors[N];
}

static void addPnPSolverCorrespondence(PnPSolver* solver,
                                       r64        X,
                                       r64        Y,
                                       r64        Z,
                                       r64        u,
                                       r64        v)
{
    solver->pws[3 * solver->numberOfCorrespondences]     = X;
    solver->pws[3 * solver->numberOfCorrespondences + 1] = Y;
    solver->pws[3 * solver->numberOfCorrespondences + 2] = Z;

    solver->us[2 * solver->numberOfCorrespondences]     = u;
    solver->us[2 * solver->numberOfCorrespondences + 1] = v;

    solver->numberOfCorrespondences++;
}

void checkPnPSolverInliers(PnPSolver* solver)
{
    solver->inliersCount = 0;

    for (int i = 0; i < solver->pointCount; i++)
    {
        cv::Point3f P3Dw = solver->mvP3Dw[i];
        cv::Point2f P2D  = solver->mvP2D[i];

        float Xc    = solver->mRi[0][0] * P3Dw.x + solver->mRi[0][1] * P3Dw.y + solver->mRi[0][2] * P3Dw.z + solver->mti[0];
        float Yc    = solver->mRi[1][0] * P3Dw.x + solver->mRi[1][1] * P3Dw.y + solver->mRi[1][2] * P3Dw.z + solver->mti[1];
        float invZc = 1 / (solver->mRi[2][0] * P3Dw.x + solver->mRi[2][1] * P3Dw.y + solver->mRi[2][2] * P3Dw.z + solver->mti[2]);

        double ue = solver->uc + solver->fu * Xc * invZc;
        double ve = solver->vc + solver->fv * Yc * invZc;

        float distX = P2D.x - ue;
        float distY = P2D.y - ve;

        float error2 = distX * distX + distY * distY;

        if (error2 < solver->mvMaxError[i])
        {
            solver->mvbInliersi[i] = true;
            solver->inliersCount++;
        }
        else
        {
            solver->mvbInliersi[i] = false;
        }
    }
}

static bool32 pnpSolverRefine(PnPSolver* solver)
{
    std::vector<int> vIndices;
    vIndices.reserve(solver->mvbBestInliers.size());

    for (size_t i = 0; i < solver->mvbBestInliers.size(); i++)
    {
        if (solver->mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    setMaximumNumberOfPnPSolverCorrespondences(solver, vIndices.size());
    solver->numberOfCorrespondences = 0;

    for (size_t i = 0; i < vIndices.size(); i++)
    {
        int idx = vIndices[i];

        addPnPSolverCorrespondence(solver,
                                   solver->mvP3Dw[idx].x,
                                   solver->mvP3Dw[idx].y,
                                   solver->mvP3Dw[idx].z,
                                   solver->mvP2D[idx].x,
                                   solver->mvP2D[idx].y);
    }

    // Compute camera pose
    computePnPSolverPose(solver, solver->mRi, solver->mti);

    // Check inliers
    checkPnPSolverInliers(solver);

    solver->mnRefinedInliers  = solver->inliersCount;
    solver->mvbRefinedInliers = solver->mvbInliersi;

    if (solver->inliersCount > solver->ransacParameters.minInliers)
    {
        cv::Mat Rcw(3, 3, CV_64F, solver->mRi);
        cv::Mat tcw(3, 1, CV_64F, solver->mti);
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        solver->mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(solver->mRefinedTcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(solver->mRefinedTcw.rowRange(0, 3).col(3));
        return true;
    }

    return false;
}

// PnPsolver::iterate
static cv::Mat solvePnP(PnPSolver*           solver,
                        i32                  nIterations,
                        bool32*              bNoMore,
                        std::vector<bool32>& vbInliers,
                        i32*                 nInliers)
{
    *bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    setMaximumNumberOfPnPSolverCorrespondences(solver,
                                               solver->ransacParameters.minSet);

    if (solver->pointCount < solver->ransacParameters.minInliers)
    {
        *bNoMore = true;
        return cv::Mat();
    }

    std::vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while (solver->globalIterationCount < solver->ransacParameters.maxIterations ||
           nCurrentIterations < nIterations)
    {
        nCurrentIterations++;
        solver->globalIterationCount++;
        solver->numberOfCorrespondences = 0;

        vAvailableIndices = solver->mvAllIndices;

        // Get min set of points
        for (short i = 0; i < solver->ransacParameters.minSet; i++)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            addPnPSolverCorrespondence(solver,
                                       solver->mvP3Dw[idx].x,
                                       solver->mvP3Dw[idx].y,
                                       solver->mvP3Dw[idx].z,
                                       solver->mvP2D[idx].x,
                                       solver->mvP2D[idx].y);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        computePnPSolverPose(solver, solver->mRi, solver->mti);

        // Check inliers
        checkPnPSolverInliers(solver);

        if (solver->inliersCount >= solver->ransacParameters.minInliers)
        {
            // If it is the best solution so far, save it
            if (solver->inliersCount > solver->bestInliersCount)
            {
                solver->mvbBestInliers   = solver->mvbInliersi;
                solver->bestInliersCount = solver->inliersCount;

                cv::Mat Rcw(3, 3, CV_64F, solver->mRi);
                cv::Mat tcw(3, 1, CV_64F, solver->mti);
                Rcw.convertTo(Rcw, CV_32F);
                tcw.convertTo(tcw, CV_32F);
                solver->mBestTcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(solver->mBestTcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(solver->mBestTcw.rowRange(0, 3).col(3));
            }

            if (pnpSolverRefine(solver))
            {
                *nInliers = solver->mnRefinedInliers;
                vbInliers = std::vector<bool32>(solver->mvpMapPointMatches.size(), false);
                for (int i = 0; i < solver->pointCount; i++)
                {
                    if (solver->mvbRefinedInliers[i])
                        vbInliers[solver->mvKeyPointIndices[i]] = true;
                }
                return solver->mRefinedTcw.clone();
            }
        }
    }

    if (solver->globalIterationCount >= solver->ransacParameters.maxIterations)
    {
        *bNoMore = true;
        if (solver->bestInliersCount >= solver->ransacParameters.minInliers)
        {
            solver->inliersCount = solver->bestInliersCount;
            vbInliers            = std::vector<bool32>(solver->mvpMapPointMatches.size(), false);
            for (int i = 0; i < solver->pointCount; i++)
            {
                if (solver->mvbBestInliers[i])
                {
                    vbInliers[solver->mvKeyPointIndices[i]] = true;
                }
            }
            return solver->mBestTcw.clone();
        }
    }

    return cv::Mat();
}