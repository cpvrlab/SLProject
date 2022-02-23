//#############################################################################
//  File:      HttpUtils.h
//  Date:      2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Luc Girod
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef HTTP_UTILS_H
#define HTTP_UTILS_H

#ifdef SL_BUILD_WITH_OPENSSL

#    ifdef _WINDOWS
#        include <winsock2.h>
#        include <ws2tcpip.h>
#    else
#        include <sys/socket.h>
#        include <netinet/in.h>
#        include <arpa/inet.h>
#    endif

#    include <vector>
#    include <string>
#    include <openssl/ssl.h>
#    include <openssl/err.h>
#    include <functional>
#    include <atomic>

using std::function;
using std::string;
using std::vector;

#    define SERVER_NOT_REACHABLE 1
#    define CANT_CREATE_DIR 2
#    define CANT_CREATE_FILE 3
#    define CONNECTION_CLOSED 4

//------------------------------------------------------------------------------
//! Multiplatform socket helper
class Socket
{
public:
    int fd;
    union
    {
        struct sockaddr_in  sa;
        struct sockaddr_in6 sa6;
    };

    socklen_t addrlen;
    bool      inUse;
    int       ipv;
#    ifdef _WINDOWS
    static WSADATA wsadata;
    static bool    initialized;
#    endif

    static bool SocketEnable()
    {
        int ret;

#    ifdef _WINDOWS
        if (!initialized)
        {
            ret = WSAStartup(MAKEWORD(2, 2), &wsadata);
            if (ret == 0)
                initialized = true;
            else
                return false;
        }
#    endif
        return true;
    }

    static void SocketDisable()
    {
#    ifdef _WINDOWS
        if (initialized)
        {
            initialized = false;
            WSACleanup();
        }
#    endif
    }

    Socket() { reset(); }

    virtual void reset();
    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual int  receive(function<int(char* data, int size)> dataCB, int max = 0);
    virtual void disconnect();
    void         interrupt() { _interrupt = true; };

protected:
    std::atomic_bool _interrupt{false};
};
//------------------------------------------------------------------------------
//! Multiplatform socket helper with encryption
class SecureSocket : public Socket
{
public:
    SecureSocket()
    {
        ssl   = nullptr;
        sslfd = -1;
        Socket::reset();
    }

    SSL* ssl;
    int  sslfd;

    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual int  receive(function<int(char* data, int size)> dataCB, int max = 0);
    virtual void disconnect();
};
//------------------------------------------------------------------------------
//! helper struct to get DNS from ip and ip from DNS
struct DNSRequest
{
    string addr;
    string hostname;
    DNSRequest(string host);
    string getAddr();
    string getHostname();
};
//------------------------------------------------------------------------------
//! HttpUtils provides networking functionality via the HTTP and HTTPS protocols
namespace HttpUtils
{
//! Class to make http get request
class GetRequest
{
public:
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

    // Read http header
    int processHttpHeaders(vector<char>& data);

    // Send Http request to server
    int send();

    // Start fetching the content (must be called after send)
    int getContent(function<int(char* data, int size)> contentCB);

    // Parse http listing and return list of directory and files
    vector<string> getListing();
};
//------------------------------------------------------------------------------
//! HTTPUtils::download provides download function for https/http with/without auth.

// Download a file chunk by chunk of 1kB.
// download is interrupt if any of the callback return non zero value.
/*!
 * @param url            url to the file / listing to download
 * @param processFile    A callback which is called when a new file will be downloaded.
 * @param writeChunk     A callback which is called for every new chunk of the current
 *                       file being downloaded.
 * @param processDir     A callback which is called when a new directory is being downloaded.
 * @param user           Username (optional) when site require http auth.
 * @param pwd            Password (optional) when site require http auth.
 * @param base           Path where the files should be saved.
 */
int download(string                                               url,
             function<int(string path, string file, size_t size)> processFile,
             function<int(char* data, int size)>                  writeChunk,
             function<int(string)>                                processDir,
             string                                               user = "",
             string                                               pwd  = "",
             string                                               base = "./");
//------------------------------------------------------------------------------
//! HTTP download function with login credentials
// download is interrupt if progress callback return non zero value.
/*!
 * @param url            url to the file / listing to download
 * @param dst            Path where the files should be saved.
 * @param user           Username (optional) when site require http auth.
 * @param pwd            Password (optional) when site require http auth.
 * @param progress       A callback which is called each 1kB downloaded for the current file.
 *                       The download stop if the returned value is not zero.
 */
int download(string                                      url,
             string                                      dst,
             string                                      user,
             string                                      pwd,
             function<int(size_t curr, size_t filesize)> progress = nullptr);
//------------------------------------------------------------------------------
//! HTTP download function without login credentials
int download(string                                      url,
             string                                      dst,
             function<int(size_t curr, size_t filesize)> progress = nullptr);

//-- return content Length of the HttpGet request
int length(string url, string user = "", string pwd = "");

}; // namespace HttpUtils
//------------------------------------------------------------------------------

#endif // SL_BUILD_WITH_OPENSSL
#endif // HTTP_UTILS_H
