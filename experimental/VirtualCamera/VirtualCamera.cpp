#include <iostream>
#include <vector>
#include "Camera.h"
#include "tools.h"
#include "Calibration.h"
#include "ViewStorage.h"


int main ()
{
    Camera c;
    std::vector<Vector3f> visible_points;
    std::vector<Vector2f> projected;
    std::vector<Vector2f> outliers;
    std::vector<Vector2f> corrects;

    ViewStorage vs;
    std::stringstream ss;

    for (int i = 0; i < 10; i++)
    {
        ss.str("");
        ss << "data/worldpoints_" << (i);
        std::string wppath = ss.str();

        ss.str("");
        ss << "data/keypoints_" << (i);
        std::string kppath = ss.str();

        visible_points.clear();
        projected.clear();

        load_view(visible_points, projected, wppath, kppath);
        vs.add_view(visible_points, projected);
    }

    /*
    srand(time(NULL));
    std::vector<Vector3f> cloud;
    cloud = generate_cloud(1000, 1, 1, 1);
    c.set_intrinsic(40, 640, 360);
    c.set_distortion(0.5, 0.4, 0., 0.);
    c.look_at(Vector3f(0, 0, -5), Vector3f(0, 1, 0), Vector3f(0, 0, 0));
    */

    Matrix3f intrinsic_matrix;
    Vector2f K;
    Vector2f P;


    cout << "try calibrate directly with opencv function (from a wrong starting fovy)" << endl;

    Calibration::init_matrix(intrinsic_matrix, P, K, 30, 640, 480);
    float error = Calibration::calibrate_opencv(intrinsic_matrix, P, K, 
                                          640, 480, 
                                          vs.points2d, 
                                          vs.points3d);

    cout << "== Final guessed camera paramters ==" << endl << endl;
    cout << "intrinsic matrix " << endl << intrinsic_matrix << endl;
    cout << "fovy : " << get_fovy(intrinsic_matrix) << endl;
    cout << "k1, k2 : " << K[0] << " " << K[1] << endl;
    cout << "p1, p2 : " << P[0] << " " << P[1] << endl << endl;
    cout << "error = " << error << endl << endl << endl;
    

    cout << "try calibrate by testing all fovy" << endl;

    error = Calibration::calibrate_brute_force(intrinsic_matrix, P, K, 640, 480, vs.points2d, vs.points3d);
    cout << "== Final guessed camera paramters ==" << endl << endl;
    cout << "intrinsic matrix " << endl << intrinsic_matrix << endl;
    cout << "fovy : " << get_fovy(intrinsic_matrix) << endl;
    cout << "k1, k2 : " << K[0] << " " << K[1] << endl;
    cout << "p1, p2 : " << P[0] << " " << P[1] << endl << endl;
    cout << "error = " << error << endl << endl << endl;

    
    //Find calibration with ransac
    //nb_iter, percent_correct, threshold, subset size
    error = 9999999;

    cout << "try calibrate with ransac" << endl;

    if (Calibration::calibrate_ransac(intrinsic_matrix, P, K, 640, 480, error, 40, 50, 20, 4, vs.points2d, vs.points3d))
    {
        cout << "== Final guessed camera paramters ==" << endl << endl;
        cout << "intrinsic matrix " << endl << intrinsic_matrix << endl;
        cout << "fovy : " << get_fovy(intrinsic_matrix) << endl;
        cout << "k1, k2 : " << K[0] << " " << K[1] << endl;
        cout << "p1, p2 : " << P[0] << " " << P[1] << endl << endl;
        cout << "error = " << error << endl << endl << endl;
    }


    return 0;
}

