mkdir BUILD_IOS
cd BUILD_IOS
cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=arm -DXCODE_CODESIGNIDENTITY="Apple Development: Marcus Hudritsch (AC3R56ZPYR)"