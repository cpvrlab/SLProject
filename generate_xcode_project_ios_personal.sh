#setup automatic app signing to enable app installation on your device:
#-you need an iPhone or an iPad
#-you need an apple id and a free apple developer account
#-you have to install xcode
#-you have to install cmake (I use dmg from the official website, uninstall all other versions before)
#-add your apple id to xcode (https://learnappmaking.com/how-to-create-a-free-apple-developer-account/)
#   -Start Xcode on your Mac
#   -Choose the Xcode → Preferences menu and navigate to the Account pane
#   -Click the +-button in the bottom-left corner and choose Apple ID
#   -Log in with your Apple ID email address and password
#-transfer your personal developer team id to cmake
#   -open the app "Keychain Access"
#   -on the top left of the window, select "login" under "Keychains" and underneath in "Category" select "Certificates"
#   -in the main window double click on your apple developer certificate (e.g. Apple Development: youremail@nothing.com (1234556))
#   -in the pop up window, copy (only) the id in the line containing "Organisational Unit". This is your personal developer team id.
#   -transfer you developer team id to cmake (e.g. -DXCODE_DEVELOPMENTTEAM="<your_id>")
#-open a terminal, navigate to this directory and run the following commands
#   chmod +x generate_xcode_project_ios_personal.sh
#   ./generate_xcode_project_ios_personal.sh
#-open the xcode project file from directory BUILD_IOS
#-Build and install the app. You will get a prompt saying: Could not launch “<app name>"
#-On your device: go to Settings/General/Device Management and select "Apple Development: <your email user>". Select "Trust "Apple Development: <your email user>""
#-Back in xcode, build and install the app.

mkdir BUILD_IOS
cd BUILD_IOS
cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_SYSTEM_PROCESSOR=arm -DXCODE_DEVELOPMENTTEAM="MZ552MQBH5"
