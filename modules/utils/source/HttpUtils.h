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
typedef int socklen_t;
#    include <winsock.h>
#else
#    include <sys/socket.h>
#    include <netinet/in.h>
#    include <arpa/inet.h>
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <functional>
#include <atomic>

using std::string;
using std::vector;
using std::function;

//------------------------------------------------------------------------------
//! ???
class Socket
{
public:
    int                fd;
    struct sockaddr_in sa;
    socklen_t          addrlen;
    bool               inUse;
#ifdef _WINDOWS
    static WSADATA wsadata;
    static bool    initialized;
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
#ifdef _WINDOWS
        if (initialized)
        {
            initialized = false;
            WSACleanup();
        }
#endif
    }

    Socket() { reset(); }

    virtual void reset();
    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(function<int(char* data, int size)> dataCB, int max = 0);
    virtual void disconnect();
    void         interrupt() { _interrupt = true; };

protected:
    std::atomic_bool _interrupt{false};
};
//------------------------------------------------------------------------------
//! ???
class SecureSocket : public Socket
{
    public:
    SecureSocket() { Socket::reset(); }

    SSL* ssl;
    int  sslfd;

    virtual int  connectTo(string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(function<int(char* data, int size)> dataCB, int max = 0);
    virtual void disconnect();
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
    int            processHttpHeaders(vector<char>& data);

    // Send Http request to server
    int            send();

    // Start fetching the content (must be called after send)
    void           getContent(function<int(char* data, int size)> contentCB);

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
}
//------------------------------------------------------------------------------