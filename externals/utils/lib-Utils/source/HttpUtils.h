//#############################################################################
//  File:      HttpUtils.h
//  Author:    Luc Girod
//  Date:      2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vector>
#include <string>
#ifdef _WINDOWS
//#    include "winsock2.h"
//-lwsock32 -lws2_32
typedef int socklen_t;
#include <winsock.h>
#else
#    include <sys/socket.h>
#    include <netinet/in.h>
#    include <arpa/inet.h>
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <functional>

using namespace std;

//------------------------------------------------------------------------------
//! ???
struct Socket
{
    int                fd;
    struct sockaddr_in sa;
    socklen_t          addrlen;
    bool               inUse;
    #ifdef _WINDOWS
    static WSADATA     wsadata;
    static bool        initialized;
    #endif

    static bool SocketEnable()
    {
        int ret;
        #ifdef _WINDOWS
        if (!initialized)
        {
            ret = WSAStartup(MAKEWORD(2, 2), &wsadata);
            if (ret == 0)
                initialized = true;
            else
                return false;
        }
        #endif
        return true;
    }

    static void SocketDisable()
    {
    }

    Socket() { reset(); }

    virtual void reset();
    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(function<void(char* data, int size)> dataCB, int max = 0);
};
//------------------------------------------------------------------------------
//! ???
struct SecureSocket : Socket
{
    SecureSocket() { Socket::reset(); }

    SSL* ssl;
    int  sslfd;

    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(function<void(char* data, int size)> dataCB, int max = 0);
};
//------------------------------------------------------------------------------
//! ???
struct DNSRequest
{
    string addr;
    string hostname;
    DNSRequest(string host);
    string getAddr();
    string getHostname();
};
//------------------------------------------------------------------------------
namespace HttpUtils
{
//! ???
struct GetRequest
{
    Socket*      s;
    vector<char> firstBytes;
    int          contentOffset;

    string request;
    string host;
    string addr;
    int    port;

    string headers;
    string version;
    string status;
    int    statusCode;
    string contentType;
    size_t contentLength;

    GetRequest(string url, string user = "", string pwd = "");
    ~GetRequest()
    {
        if (s) { delete s; }
    }

    int            processHttpHeaders(vector<char>& data);
    int            send();
    void           getContent(function<void(char* data, int size)> contentCB);
    vector<string> getListing();
};
//------------------------------------------------------------------------------
//! ???
int download(string                                               url,
             function<int(string path, string file, size_t size)> processFile,
             function<int(char* data, int size)>                  writeChunk,
             function<int(string)>                                processDir,
             string                                               user = "",
             string                                               pwd  = "",
             string                                               base = "./");
//------------------------------------------------------------------------------
//! HTTP download function with login credentials
int download(string                                       url,
             string                                       dst,
             string                                       user,
             string                                       pwd,
             function<void(size_t curr, size_t filesize)> progress = nullptr);
//------------------------------------------------------------------------------
//! HTTP download function without login credentials
int download(string                                       url,
             string                                       dst,
             function<void(size_t curr, size_t filesize)> progress = nullptr);
}
//------------------------------------------------------------------------------
