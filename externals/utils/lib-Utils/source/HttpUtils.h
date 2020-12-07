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
    virtual int               connectTo(std::string ip, int port);
    virtual int               sendData(const char* data, size_t size);
    virtual std::vector<char> recieve();
};

struct SecureSocket : Socket
{
    SecureSocket() { Socket::reset(); }

    SSL* ssl;
    int  sslfd;

    virtual int               connectTo(std::string ip, int port);
    virtual int               sendData(const char* data, size_t size);
    virtual std::vector<char> recieve();
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
        bool        isSecure;
        std::string request;
        std::string host;
        std::string addr;
    
        std::string       headers;
        std::string       version;
        std::string       status;
        int               statusCode;
        std::string       contentType;
        size_t            contentLength;
        std::vector<char> content;
    
        GetRequest(std::string url, std::string user = "", std::string pwd = "");

        int                      send();
        std::vector<char>        getContent();
        std::vector<std::string> getListing();
    };

    void download(std::string                                         url,
                  std::function<void(std::string, std::vector<char>)> f,
                  std::function<void(std::string)>                    subdir,
                  std::string                                         user = "",
                  std::string                                         pwd  = "",
                  std::string                                         base = "./");

    void download(std::string url, std::string dst, std::string user = "", std::string pwd = "");

}
