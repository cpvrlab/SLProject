#include <iostream>
#include <stdio.h>
#include "tools.h"
#include "app.h"
#include "gui.h"

int main(int argc, char** argv)
{
    App app;

    if (argc < 3)
    {
        std::cout << "Usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        exit(1);
    }

    app.name = "Best tool ever made!";
    app.closeup_left = "closeup left";
    app.closeup_right = "closeup right";
    app.select_radius = 10;
    app.local_idx = 0;
    app.right_idx = 0;
    app.left_idx = 0;
    app.method = STOCK_ORBSLAM;

    app.image1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    app.image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if (!app.image1.data || !app.image2.data)
    {
        std::cout << "Usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        std::cout << "Can't open images" << std::endl;
        exit(1);
    }

    app_prepare(app);
    
    start_gui(app);

    return 0;
}


