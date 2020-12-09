#include <iostream>
#include <cstring>
#include <resolv.h>
#include <errno.h>
#include <sstream>
#include <netdb.h>
#include <HttpUtils.h>
#include <Utils.h>
using namespace std;

void Socket::reset()
{
    fd    = 0;
    inUse = false;
    memset(&sa, 0, sizeof(sa));
}

int Socket::connectTo(std::string ip, int port)
{
    memset(&sa, 0, sizeof(sa));
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = inet_addr(ip.c_str());
    sa.sin_port        = htons(port);
    addrlen            = sizeof(sa);

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!fd)
    {
        std::cerr  << "Error creating socket.\n" << std::endl;
        return -1;
    }

    if (connect(fd, (struct sockaddr*)&sa, addrlen))
    {
        std::cerr  << "Error connecting to server.\n" << std::endl;
        return -1;
    }
    inUse = true;
    return 0;
}

int Socket::sendData(const char* data, size_t size)
{
    int len = send(fd, data, size, 0);
    if (len < 0)
        return -1;
    return 0;
}

#define BUFFER_SIZE 1000
void Socket::receive(std::function<void(char * data, int size)> dataCB, int max)
{
    int n = 0;
    int len;
    char buf[BUFFER_SIZE];
    do
    {
        if (max != 0 && max - n <= BUFFER_SIZE)
        {
            len = recv(fd, buf, max - n, 0);
            dataCB(buf, len);
            return;
        }
        len = recv(fd, buf, BUFFER_SIZE, 0);
        n = n + len;
        dataCB(buf, len);
    }
    while (len > 0);
}

int SecureSocket::connectTo(std::string ip, int port)
{
    Socket::connectTo(ip, port);
    SSL_load_error_strings();
    const SSL_METHOD* meth = TLS_client_method();
    SSL_CTX*          ctx  = SSL_CTX_new(meth);
    ssl                    = SSL_new(ctx);
    if (!ssl)
    {
        std::cerr << "Error creating SSL.\n"
                  << std::endl;
        return -1;
    }
    sslfd = SSL_get_fd(ssl);
    SSL_set_fd(ssl, fd);
    int err = SSL_connect(ssl);
    if (err <= 0)
    {
        std::cerr << "Error creating SSL connection.  err= " << err << std::endl;
        return -1;
    }

    return 0;
}

int SecureSocket::sendData(const char* data, size_t size)
{
    int len = SSL_write(ssl, data, size);
    if (len < 0)
    {
        int err = SSL_get_error(ssl, len);
        switch (err)
        {
            case SSL_ERROR_WANT_WRITE:
                return 0;
            case SSL_ERROR_WANT_READ:
                return 0;
            case SSL_ERROR_ZERO_RETURN:
            case SSL_ERROR_SYSCALL:
            case SSL_ERROR_SSL:
            default:
                return -1;
        }
    }
    return 0;
}

void SecureSocket::receive(std::function<void(char * data, int size)> dataCB, int max)
{
    int  len;
    int  n = 0;
    char buf[BUFFER_SIZE];
    do
    {
        if (max != 0 && max - n <= BUFFER_SIZE)
        {
            len = SSL_read(ssl, buf, max - n);
            dataCB(buf, len);
            return;
        }
        len = SSL_read(ssl, buf, BUFFER_SIZE);
        n += len;
        dataCB(buf, len);
    }
    while (len > 0);
}

DNSRequest::DNSRequest(std::string host)
{
    bool hostIsAddr = true;
    for (int c : host)
    {
        if (!isdigit(c) && c != '.') { hostIsAddr = false; }
        break;
    }

    if (hostIsAddr)
    {
        struct sockaddr_in sa;
        sa.sin_family      = AF_INET;
        sa.sin_addr.s_addr = inet_addr(host.c_str());

        struct hostent* h = gethostbyaddr(&sa.sin_addr, sizeof(sa.sin_addr), sa.sin_family);
        if (h != nullptr && h->h_length > 0)
        {
            hostname = std::string(h->h_name);
        }
        addr = host;
    }
    else
    {
        struct hostent* h = gethostbyname(host.c_str());
        if (h != nullptr && h->h_length > 0)
        {
            addr = std::string(inet_ntoa( (struct in_addr) *((struct in_addr *) h->h_addr_list[0])));
        }
        hostname = host;
    }
}

std::string DNSRequest::getAddr()
{
    return addr;
}

std::string DNSRequest::getHostname()
{
    return hostname;
}

static std::string base64(const std::string data)
{
    static constexpr char sEncodingTable[] = {
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

    size_t      in_len  = data.size();
    size_t      out_len = 4 * ((in_len + 2) / 3);
    std::string ret(out_len, '\0');
    size_t      i;
    char*       p = const_cast<char*>(ret.c_str());

    for (i = 0; i < in_len - 2; i += 3)
    {
        *p++ = sEncodingTable[(data[i] >> 2) & 0x3F];
        *p++ = sEncodingTable[((data[i] & 0x3) << 4) | ((int)(data[i + 1] & 0xF0) >> 4)];
        *p++ = sEncodingTable[((data[i + 1] & 0xF) << 2) | ((int)(data[i + 2] & 0xC0) >> 6)];
        *p++ = sEncodingTable[data[i + 2] & 0x3F];
    }
    if (i < in_len)
    {
        *p++ = sEncodingTable[(data[i] >> 2) & 0x3F];
        if (i == (in_len - 1))
        {
            *p++ = sEncodingTable[((data[i] & 0x3) << 4)];
            *p++ = '=';
        }
        else
        {
            *p++ = sEncodingTable[((data[i] & 0x3) << 4) | ((int)(data[i + 1] & 0xF0) >> 4)];
            *p++ = sEncodingTable[((data[i + 1] & 0xF) << 2)];
        }
        *p++ = '=';
    }

    return ret;
}

static void parseURL(std::string url, std::string &host, std::string &path, bool &useTLS)
{
    host     = "";
    useTLS = false;

    std::string dir = "/";
    std::string tmp;
    size_t      offset = 0;

    if (url.find("https://") != std::string::npos)
    {
        useTLS = true;
        offset   = 8;
    }
    else if (url.find("http://") != std::string::npos)
    {
        offset = 7;
    }

    size_t pos = url.find("/", offset);
    if (pos != std::string::npos)
    {
        path = url.substr(pos, url.length() - pos);
        host = url.substr(offset, pos - offset);
    }
    else
        host = url.substr(offset);
}

HttpUtils::GetRequest::GetRequest(std::string url, std::string user, std::string pwd)
{
    std::string path;
    bool        isSecure;

    parseURL(url, host, path, isSecure);

    DNSRequest dns(host);
    host = dns.getHostname();
    addr = dns.getAddr();

    request = "GET " + path + " HTTP/1.1\r\n";

    if (!user.empty())
    {
        request = request + "Authorization: Basic " + base64(user + ":" + pwd) + "\r\n";
    }
    request = request + "Host: " + host + "\r\n\r\n";

    if (isSecure)
    {
        s = new SecureSocket();
        port = 443;
    }
    else
    {
        s = new Socket();
        port = 80;
    }
}

int HttpUtils::GetRequest::processHttpHeaders(std::vector<char> &data)
{
    std::string h = std::string(data.begin(), data.begin() + (data.size() > 1000 ? 1000 : data.size()));
    size_t contentPos = h.find("\r\n\r\n");
    if (contentPos == std::string::npos)
    {
        std::cout << "Invalid http response\n" << std::endl;
        return -1;
    }
    headers = std::string(data.begin(), data.begin() + contentPos);

    size_t pos = headers.find("HTTP");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos, headers.find("\r", pos) - pos);
        size_t codeIdx;
        for (codeIdx = 0; codeIdx < str.length() && !isdigit(str.at(codeIdx)); codeIdx++);

        if (codeIdx == str.length())
        {
            std::cout << "Invalid http response\n" << std::endl;
            return -1;
        }

        size_t endCodeIdx = str.find(" ", codeIdx);

        statusCode = stoi(str.substr(codeIdx, endCodeIdx - codeIdx));
        status = str.substr(endCodeIdx);
        status = Utils::trimString(status, " ");
    }

    pos = headers.find("Content-Length:");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos + 16, headers.find("\r", pos+16) - pos - 16);
        contentLength = stoi(Utils::trimString(str, " "));
    }

    pos = headers.find("Content-Type:");
    if (pos != std::string::npos)
    {
        std::string str = headers.substr(pos + 13, headers.find("\r", pos+13) - pos - 13);
        contentType = Utils::trimString(str, " ");
    }
    return contentPos;
}

int HttpUtils::GetRequest::send()
{
    if (s->connectTo(addr, port) < 0)
    {
        std::cerr << "could not connect\n" << std::endl;
        return -1;
    }

    s->sendData(request.c_str(), request.length() + 1);

    std::vector<char> * v = &firstBytes;
    s->receive([v](char* buf, int size) -> void {
                v->reserve(v->size() + size);
                copy(&buf[0], &buf[size], back_inserter(*v));
            },
            1000);

    contentOffset = processHttpHeaders(firstBytes);
    return 0;
}

void HttpUtils::GetRequest::getContent(std::function<void(char * buf, int size)> contentCB)
{
    if (contentOffset < firstBytes.size())
        contentCB(firstBytes.data() + contentOffset, firstBytes.size() - contentOffset);

    s->receive(contentCB);
}

std::vector<std::string> HttpUtils::GetRequest::getListing()
{
    std::vector<char> content;
    getContent([&content] (char * buf, int size)-> void { 
        content.reserve(content.size() + size);
        copy(&buf[0], &buf[size], back_inserter(content));
        });

    std::string c = std::string(content.data());
    std::vector<string> listing;

    size_t pos = 0;
    size_t end = 0;
    while(1)
    {
        pos = c.find("<", pos);
        end = c.find(">", pos)+1;
        if (pos == std::string::npos || end == std::string::npos)
            break;

        std::string token = c.substr(pos+1, end - pos - 2);
        token = Utils::trimString(token, " ");

        if (std::string(token.begin(), token.begin()+2) == "a ")
        {
            size_t href = token.find("href");
            if (href != std::string::npos)
            {
                href = token.find("\"", href);
                size_t hrefend = token.find("\"", href + 1);
                if (token.find("?") == std::string::npos)
                {
                    token = token.substr(href+1, hrefend - href -1);
                    if (token == "../" || token == ".")
                    {
                        pos = end;
                        continue;
                    }
                    listing.push_back(token);
                }
            }
        }
        pos = end;
    }
    return listing;
}

void HttpUtils::download(std::string                                                          url,
                         std::function<void(std::string path, std::string name, size_t size)> processFile,
                         std::function<void(char* data, int size)>                            writeChunk,
                         std::function<void(std::string)>                                     processDir,
                         std::string                                                          user,
                         std::string                                                          pwd,
                         std::string                                                          base)
{

    HttpUtils::GetRequest req = HttpUtils::GetRequest(url, user, pwd);
    req.send();
    base = Utils::unifySlashes(base);

    if (req.contentType == "text/html")
    {
        if (url.back() != '/')
            url = url + "/";

        processDir(base);

        std::vector<std::string> listing = req.getListing();
        for (std::string str : listing)
        {   
            if (str.at(0) != '/')
                download(url + str, processFile, writeChunk, processDir, user, pwd, base + str);
        }
    }
    else
    {
        std::string file = url.substr(url.rfind("/") + 1);

        int possibleSplit = base.rfind(file);
        if (possibleSplit != std::string::npos && base.size() - possibleSplit -1 == file.size())
            base = base.substr(0, possibleSplit);

        base = Utils::unifySlashes(base);

        processFile(base, file, req.contentLength);
        req.getContent(writeChunk);
    }
}

void HttpUtils::download(std::string url, std::string dst, std::string user, std::string pwd, std::function<void(size_t curr, size_t filesize)> progress)
{
    std::ofstream fs;
    size_t        totalBytes = 0;

    download(
      url,
      [&fs, &totalBytes](std::string path, std::string file, size_t size) -> void {
          fs.open(path + file, ios::out | ios::binary);
          totalBytes = size;
      },
      [&fs, progress, &totalBytes](char* data, int size) -> void {
          if (size > 0)
          {
              fs.write(data, size);
              if (progress)
                  progress(size, totalBytes);
          }
          else
              fs.close();
      },
      [](std::string dir) -> void {
          Utils::makeDir(dir);
      },
      user,
      pwd,
      dst);
}

void HttpUtils::download(std::string url, std::string dst, std::function<void(size_t curr, size_t filesize)> progress)
{
    download(url, dst, "", "", progress);
}
