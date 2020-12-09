#include <vector>
#include <string>
#ifdef _WINDOWS
    #include "winsock2.h"
    //-lwsock32 -lws2_32
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <functional>

struct Socket
{
    int                fd;
    struct sockaddr_in sa;
    socklen_t          addrlen;
    bool               inUse;

    Socket() { reset(); }

    virtual void reset();
    virtual int  connectTo(std::string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(std::function<void(char * data, int size)> dataCB, int max = 0);
};

struct SecureSocket : Socket
{
    SecureSocket() { Socket::reset(); }

    SSL* ssl;
    int  sslfd;

    virtual int  connectTo(std::string ip, int port);
    virtual int  sendData(const char* data, size_t size);
    virtual void receive(std::function<void(char * data, int size)> dataCB, int max = 0);
};

struct DNSRequest
{
    std::string addr;
    std::string hostname;
    DNSRequest(std::string host);
    std::string getAddr();
    std::string getHostname();
};

namespace HttpUtils
{
    struct GetRequest
    {
        Socket*           s;
        std::vector<char> firstBytes;
        int               contentOffset;

        std::string request;
        std::string host;
        std::string addr;
        int         port;

        std::string       headers;
        std::string       version;
        std::string       status;
        int               statusCode;
        std::string       contentType;
        size_t            contentLength;
    
        GetRequest(std::string url, std::string user = "", std::string pwd = "");
        ~GetRequest() { if (s) {delete s; } }

        int                      processHttpHeaders(std::vector<char>& data);
        int                      send();
        void                     getContent(std::function<void(char* data, int size)> contentCB);
        std::vector<std::string> getListing();
    };

    void download(std::string                                                          url,
                  std::function<void(std::string path, std::string file, size_t size)> processFile,
                  std::function<void(char* data, int size)>                            writeChunk,
                  std::function<void(std::string)>                                     processDir,
                  std::string                                                          user = "",
                  std::string                                                          pwd  = "",
                  std::string                                                          base = "./");

    void download(std::string url, std::string dst, std::string user, std::string pwd, std::function<void(size_t curr, size_t filesize)> progress = nullptr);
    void download(std::string url, std::string dst, std::function<void(size_t curr, size_t filesize)> progress = nullptr);
}
