cd ..
mkdir build_ios
cd build_ios
cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=arm64 -DXCODE_CODESIGNIDENTITY="Apple Development: Marcus Hudritsch (AC3R56ZPYR)"
