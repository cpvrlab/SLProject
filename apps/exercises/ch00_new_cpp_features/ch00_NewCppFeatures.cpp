
/////////////////////////////////////////////////////////////////
// C++ 11, 14 & 17 Standard Demos
// Resources for C++ 11, 14 & 17 features:
// http://stroustrup.com/C++11FAQ.html
// http://wiki.apache.org/stdcxx/C++0xCompilerSupport
// http://channel9.msdn.com/Events/GoingNative/GoingNative-2012
// http://en.wikipedia.org/wiki/C++11
// http://herbsutter.com/elements-of-modern-c-style/
// http://thbecker.net/articles/rvalue_references/section_01.html
// http://www.umich.edu/~eecs381/handouts/C++11_shared_ptr.pdf
// https://www.codingame.com/playgrounds/2205/7-features-of-c17-that-will-simplify-your-code/introduction
// https://www.codingame.com/playgrounds/5659/c17-filesystem
/////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <random>
#include <functional>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#ifndef __APPLE__
#    include <execution>
#endif

using std::cout;
using std::endl;
using std::function;
using std::map;
using std::move;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

using namespace std::chrono;
using namespace std::placeholders;

//-----------------------------------------------------------------------------
class A
{
public:
    A() : A(42) {} // New C++11: Delegating constructors
    A(int in_a) : a(in_a) { cout << "A(" << in_a << ")" << endl; }
    virtual ~A() { cout << "~A()" << endl; }

    int          getA() { return a; }
    virtual void f1() { cout << "f1();" << endl; }

    // New C++11: keyword final to avoid overriding
    virtual void f2() final { cout << "f2();" << endl; }

private:
    int                     a      = 4;                  // New C++11: in-class member initializers
    string                  s      = R"(C:\Afile1.txt)"; // New C++11: string literals R"(...)"
    inline static const int sValue = 777;                // New C++17: inline static initialization
};
//-----------------------------------------------------------------------------
class B : public A
{
public:
    // int f1() override {;} // New: Error on wrong override (this was only a warning before)
    // void f2(){;}          // New: Error because overriding f2 is not allowed
};
//-----------------------------------------------------------------------------
class C
{
};
//-----------------------------------------------------------------------------
int sum(int a, int b) { return a + b; }
//-----------------------------------------------------------------------------
float divide(float a, float b) { return a / b; }
//-----------------------------------------------------------------------------
struct Vector3
{
    float x, y, z;
    Vector3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    // New Cpp17 attribute nodiscard leads to a warning if a caller ignores the return value
    [[nodiscard]] float length() const { return sqrt(x * x + y * y + z * z); }
};
//-----------------------------------------------------------------------------
Vector3 normalize(const Vector3& v)
{
    float inv_len = 1.0f / v.length();
    return {v.x * inv_len, v.y * inv_len, v.z * inv_len};
}
//-----------------------------------------------------------------------------

// Forward declarations
void new_rvalue_references();
void new_variadic_templates();
void new_uniform_intializers();
void new_lambda_expressions();
void new_type_deduction();
void new_basic_types_and_type_traits();
void new_functional();
void new_threading();
void new_random_generators();
void new_smart_pointers();
void new_const_expression();
void new_userdefined_literals();
void new_if_switch_statement();
void new_structured_binding();
void new_filesystem();
void new_parallel_algorithms();

///////////////////////////////////////////////////////////////////////////////
int main()
{
    new_rvalue_references();
    new_variadic_templates();
    new_uniform_intializers();
    new_lambda_expressions();
    new_type_deduction();
    new_basic_types_and_type_traits();
    new_functional();
    new_threading();
    new_random_generators();
    new_smart_pointers();
    new_const_expression();
    new_userdefined_literals();
    new_if_switch_statement();
    new_structured_binding();
#ifndef __APPLE__
    new_filesystem();
    new_parallel_algorithms();
#endif

    return 0;
}
///////////////////////////////////////////////////////////////////////////////

int  myGlobalInt;
int& getMyGlobalInt() { return myGlobalInt; }

//-----------------------------------------------------------------------------
// Demo class for an int array with C++11 move semantics
class MyVector
{
public:
    // default constructor produces a moderately sized Array
    // The C++11 specifier explicit avoid the implicit conversion call
    explicit MyVector(int n = 10) : _data(new int[n]), _size(n) {}

    // copy constructor
    MyVector(const MyVector& other) : _data(new int[other._size]), _size(other._size)
    {
        for (int i = 0; i < _size; ++i)
            _data[i] = other._data[i];
    }

    // move constructor
    MyVector(MyVector&& other) noexcept : _data(other._data), _size(other._size)
    { // Release the data pointer from the source object so that
      // the destructor does not free the memory multiple times.
        other._data = nullptr;
        other._size = 0;
    }

    // destructor
    ~MyVector() { delete[] _data; }

    // copy assignment operator
    MyVector& operator=(MyVector& other)
    {
        if (this != &other)
        {
            delete[] _data; // Free the existing resource.
            _size = other._size;
            _data = new int[_size];
            for (int i = 0; i < _size; ++i)
                _data[i] = other._data[i];
        }
        return *this;
    }

    // move assignment operator
    MyVector& operator=(MyVector&& other) noexcept
    {
        if (this != &other)
        {
            delete[] _data; // Free the existing resource.

            // Copy the data pointer and its length from the source object.
            _data = other._data;
            _size = other._size;

            // Release the data pointer from the source object so that
            // the destructor does not free the memory multiple times.
            other._data = nullptr;
            other._size = 0;
        }
        return *this;
    }

private:
    int* _data;
    int  _size;
};

// Demo function that returns an expensive rvalue
MyVector createABigArray()
{
    MyVector am(1000);
    return am;
}

///////////////////////////////////////////////////////////////////////////////
void new_rvalue_references()
{
    high_resolution_clock::time_point t1, t2;

    // This looks simple on the first look and is complex to understand when you read more.
    // What are lvalue and rvalues?
    // lvalue - is an expression that occupies an identifiable memory location
    // rvalue - is an expression that is not an lvalue.

    // Most variables in C++ code are lvalues.
    // C++ references are lvalue references.

    // The best additional resources I found are:
    // http://thbecker.net/articles/rvalue_references/section_01.html

    // lvalue examples:
    int      i   = 42; // i is an lvalue
    int*     pi  = &i; // i & pi are an lvalues
    int&     lri = i;  // lri is a reference and an lvalue and can reference only an lvalue
    MyVector a1(1000); // a1 is an instance of class MyVector and therefore an lvalue

    // rvalue examples:
    int j = 42;        // 42 is an rvalue (you can't take the address of it)
    int x = j + 2;     // (i+2) is an rvalue
    int c = sum(1, 2); // sum(1,2) is an rvalue (you can't take the address of an rvalue)

    // let's get weired:
    int& rI          = getMyGlobalInt();  // getMyGlobalInt() is not an rvalue!
    getMyGlobalInt() = 42;                // getMyGlobalInt() is an lvalue because it returns a reference
    int* pMGI        = &getMyGlobalInt(); // ok, getMyGlobalInt() is an lvalue, I can get the adress!

    // Now new in C++11 are rvalue references with a && postfix. They can be bound only to rvalues
    // Don't use them as such.
    int&& rri = 42;

    cout << "\nNew move constructor with rvalue references:------------------------\n";
    t1 = high_resolution_clock::now();
    MyVector a2(a1); // copy constructor
    t2 = high_resolution_clock::now();
    cout << "Time used with copy: " << duration_cast<nanoseconds>(t2 - t1).count() << "ns.\n";

    t1 = high_resolution_clock::now();
    MyVector a3(move(a1)); // move constructor (move() creates an rvalue reference)
    t2 = high_resolution_clock::now();
    // Don't access a1 anymore
    cout << "Time used with move: " << duration_cast<nanoseconds>(t2 - t1).count() << "ns.\n";

    cout << "\nNew move assignment with rvalue references:------------------------\n";
    t1 = high_resolution_clock::now();
    MyVector a4;
    a4 = a2; // copy assignment
    t2 = high_resolution_clock::now();
    cout << "Time used with copy: " << duration_cast<nanoseconds>(t2 - t1).count() << "ns.\n";

    t1 = high_resolution_clock::now();
    MyVector a5;
    a5 = move(a4); // move assignment (move() creates an rvalue reference)
    t2 = high_resolution_clock::now();
    // Don't access a4 anymore
    cout << "Time used with move: " << duration_cast<nanoseconds>(t2 - t1).count() << "ns.\n";

    cout << "\nNew move assignment with rvalue references:------------------------\n";
    MyVector a6;
    t1 = high_resolution_clock::now();
    a6 = createABigArray(); // assignment of an rvalue with move semantics
    t2 = high_resolution_clock::now();
    cout << "Time used with move: " << duration_cast<nanoseconds>(t2 - t1).count() << "ns.\n";

    cout << "\nMove test: --------------------------------------------------------\n";
    string      a = " a long long string";
    string      b;
    const char* pa1 = a.c_str();
    const char* pb1 = b.c_str();
    cout << "a before the move: " << std::hex << pa1 << ", " << pa1 << endl;
    cout << "b before the move: " << std::hex << pb1 << ", " << pb1 << endl;
    b               = move(a);
    const char* pa2 = a.c_str();
    const char* pb2 = b.c_str();
    cout << "a after  the move: " << std::hex << pa2 << ", " << pa2 << endl;
    cout << "b after  the move: " << std::hex << pb2 << ", " << pb2 << endl;
    if (pa1 == pa2)
        cout << "pa points to OLD memory" << endl;
    else
        cout << "pa points to NEW memory" << endl;
    if (pb1 == pb2)
        cout << "pb points to OLD memory" << endl;
    else
        cout << "pb points to NEW memory" << endl;
    if (pb2 == pa1)
        cout << "pb points to old pa" << endl;
    else
        cout << "pb points to NEW memory" << endl;

    // Results in n 2015     2016     2017     2018     2020
    // Linux GCC    OLD/OLD  OLD/OLD  OLD/OLD  NEW/NEW  NEW/NEW
    // Mac Clang    NEW/NEW  OLD/OLD  OLD/OLD  OLD/OLD  OLD/OLD
    // Win VSC++    NEW/NEW  NEW/NEW  NEW/NEW  NEW/NEW  NEW/NEW
}
///////////////////////////////////////////////////////////////////////////////

// C++11: Variadic templates. The first add is needed to stop the recursion
template<typename T>
T addCpp11(T v)
{
    return v;
}
template<typename T, typename... Args>
T addCpp11(T first, Args... args)
{
    return first + addCpp11(args...);
}

// C++17: A lot simpler, so forget the C++11 version
template<typename... Args>
auto addCpp17(Args... args)
{
    return (args + ...);
}

///////////////////////////////////////////////////////////////////////////////
void new_variadic_templates()
{
    // Variadic template extend template with the variable number of parameters (...)
    cout << "\nVariadic templates:------------------------------------------------\n";
    string s1 = "x", s2 = "y", s3 = "z";
    cout << "addCpp11(1,2,3,4)  = " << addCpp11(1, 2, 3, 4) << endl;
    cout << "addCpp11(s1,s2,s3) = " << addCpp11(s1, s2, s3) << endl;
    cout << "addCpp17(1,2,3,4)  = " << addCpp17(1, 2, 3, 4) << endl;
    cout << "addCpp17(s1,s2,s3) = " << addCpp17(s1, s2, s3) << endl;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_uniform_intializers()
{
    cout << "\nUniform initializers:----------------------------------------------\n";
    int                 ai[] = {1, 2, 3, 4, 5};             // array init was possible before
    vector<int>         vi   = {10, 60, -10, 20, -30};      // vector init works now the same way
    Vector3             v1(1.f, 0.f, 0.f);                  // constructor of Vector3
    Vector3             v2{1.f, 0.f, 0.f};                  // init with {} works now the same way
    Vector3             vn    = normalize({.5f, .5f, .5f}); // Passing a Vector3 to normalize
    map<string, string> email = {{"Marcus", "marcus.hudritsch@bfh.ch"},
                                 {"Urs", "urs.kuenzler@bfh.ch"}};
}
///////////////////////////////////////////////////////////////////////////////

struct absIntCompair
{
    bool operator()(const int a, int b) { return abs(a) < abs(b); }
};

///////////////////////////////////////////////////////////////////////////////
void new_lambda_expressions()
{
    cout << "\nLamdas: Simple quick function:-------------------------------------\n";
    auto HelloWorld = []()
    { cout << "Hello world"; };
    HelloWorld(); // now call the function

    cout << "\nLamdas: Old style abs. sort with functor object:\n";
    vector<int> vi = {10, 60, -10, 20, -30}; // vector init works now the same way
    sort(vi.begin(), vi.end(), absIntCompair());
    for (auto i : vi) cout << i << ",";

    cout << "\nNew style abs. sort with lambda expression:\n";
    sort(vi.begin(), vi.end(), [](int a, int b)
         { return abs(a) < abs(b); });
    for (auto i : vi) cout << i << ",";

    auto fsum1 = [](int a, int b)
    { return a + b; };
    cout << "\nLambda function object with 2 int: fsum1(4,2) = " << fsum1(4, 2);

    int  a     = 4;
    auto func1 = [&](int b)
    {a++; return a + b; }; // a is accessed by reference
    cout << "\nLambda function w. captured variable by reference:     func1(2) = " << func1(2) << ", a=" << a;

    auto func2 = [=](int b)
    { return a + b; }; // a is accessed by value
    cout << "\nLambda function w. captured variable by value:         func2(2) = " << func2(2) << ", a=" << a;

    auto func3 = [=](int b) mutable
    {a++; return a + b; }; // a is accessed by mutable value
    cout << "\nLambda function w. captured variable by mutable value: func3(2) = " << func3(2) << ", a=" << a;

    auto func4 = []
    { printf("\nLambda functions can call any other function."); };
    func4();

    function<int(int)> func5 = [&func5](int n)
    { return n <= 1 ? 1 : n * func5(n - 1); };
    cout << "\nLambda function can be called recursively: func5(4) = " << func5(4) << endl;

    cout << "\nC++14: Generic Lambda function with all auto type deduction:" << endl;
    vector<int>    ivec  = {1, 2, 3, 4};
    vector<string> svec  = {"red", "green", "blue"};
    auto           adder = [](auto op1, auto op2)
    { return op1 + op2; };
    cout << "int    result: " << accumulate(ivec.begin(), ivec.end(), 0, adder) << endl;
    cout << "string result: " << accumulate(svec.begin(), svec.end(), string(""), adder) << endl;
}
///////////////////////////////////////////////////////////////////////////////

// C++14 allows auto as return type
auto funcReturnsAuto() { return 0; }

///////////////////////////////////////////////////////////////////////////////
void new_type_deduction()
{
    // New: auto type for automatic static type deduction at compile time
    int                 arrayi[] = {1, 2, 3, 4, 5};
    vector<int>         vectori  = {10, 60, -10, 20, -30};
    map<string, string> mapemail = {{"Marcus", "marcus.hudritsch@bfh.ch"},
                                    {"Urs", "urs.kuenzler@bfh.ch"}};

    cout << "\nauto:--------------------------------------------------------------\n";
    auto i1 = 1;
    cout << "This is an int: " << i1 << endl;
    auto f1 = 1.1f;
    cout << "This is a float: " << f1 << endl;
    auto d1 = 1.1;
    cout << "This is a double: " << d1 << endl;
    auto* b1 = new B();
    cout << "This is an instance of B:" << b1->getA() << endl;
    auto e1 = mapemail.begin();
    cout << "This is an email address: " << (*e1).second << endl;

    cout << "\nC++14 auto as return type:-----------------------------------------\n";
    auto a = funcReturnsAuto();
    cout << "funcReturnsAuto return: " << typeid(a).name() << endl;

    // New: decltype for type deduction from a variable
    cout << "\ndecltype:----------------------------------------------------------\n";
    decltype(i1) i2 = 1;
    cout << "This is an int: " << i2 << endl;
    decltype(f1 * i1) f2 = i1 * f1;
    cout << "This is an float: " << f2 << endl;

    // Old style STL for loop
    cout << "\nOld style for loop:------------------------------------------------\n";
    for (map<string, string>::iterator i = mapemail.begin(); i != mapemail.end(); ++i)
        cout << (*i).first << ": " << (*i).second << endl;

    // New: range-for statement
    cout << "\nNew style range-for loops:-----------------------------------------\n";
    for (auto i : arrayi) cout << i << ",";
    cout << endl;
    for (auto i : mapemail) cout << i.first << ": " << i.second << ",";
    cout << endl;
    for (auto i : vectori) cout << i << ",";
    cout << endl;
    for (auto i : {1, 2, 3, 5, 8, 13, 21, 34}) cout << i << ",";
    cout << endl;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_basic_types_and_type_traits()
{
    // New standard data types are typedefs in stdint.h
    cout << "\nNew int types:-----------------------------------------------------\n";
    int8_t i8 = -1;
    cout << "This is an  8 bit integer:          " << i8 << endl;
    uint8_t ui8 = 255;
    cout << "This is an  8 bit unsigned integer: " << ui8 << endl;
    int16_t i16 = -1000;
    cout << "This is an 16 bit integer:          " << i16 << endl;
    uint16_t iu16 = 1000;
    cout << "This is an 16 bit unsigned integer: " << iu16 << endl;
    int32_t i32 = 1000;
    cout << "This is an 32 bit integer:          " << i32 << endl;
    uint32_t ui32 = 1000;
    cout << "This is an 32 bit unsigned integer: " << ui32 << endl;
    int64_t i64 = 1000;
    cout << "This is an 64 bit integer:          " << i64 << endl;
    uint64_t ui64 = 1000;
    cout << "This is an 64 bit unsigned integer: " << ui64 << endl;

    // New type traits for RTTI (there are many more)
    cout << "\nType traits:-------------------------------------------------------\n";
    cout << "has_virtual_destructor<int>::value:" << std::has_virtual_destructor<int>::value << endl;
    cout << "is_polymorphic<int>::value: " << std::is_polymorphic<int>::value << endl;
    cout << "is_floating_point<int>: " << std::is_floating_point<int>::value << endl;
    cout << "is_polymorphic<A>::value: " << std::is_polymorphic<A>::value << endl;
    cout << "is_polymorphic<C>::value: " << std::is_polymorphic<C>::value << endl;

    typedef int mytype[][24][60];
    cout << "typedef int mytype[][24][60]:" << endl;
    cout << "extent<mytype,0>::value: " << std::extent<mytype, 0>::value << endl;
    cout << "extent<mytype,1>::value: " << std::extent<mytype, 1>::value << endl;
    cout << "extent<mytype,2>::value: " << std::extent<mytype, 2>::value << endl;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_functional()
{
    // New: function object from a C-function
    cout << "\nOld C-function pointer:--------------------------------------------\n";
    typedef int (*ptFunc)(int, int);
    ptFunc fsumC = &sum;
    cout << "Sum C-function pointer: fsumC(4,2) = " << fsumC(4, 2) << endl;

    cout << "\nNew Function object on C-functions:--------------------------------\n";
    function<int(int, int)> fsum = &sum;
    cout << "Sum function object: fsum(4,2) = " << fsum(4, 2) << endl;

    // New: function object from a C-function using bind
    cout << "\nFunction object on C-functions using bind with placeholders:-------\n";
    function<float(float, float)> inv_div = std::bind(divide, _2, _1);
    cout << "1/6: inv_div(6,1) =" << inv_div(6, 1) << endl;
    cout << "2/6: inv_div(6,2) =" << inv_div(6, 2) << endl;
    cout << "3/6: inv_div(6,3) =" << inv_div(6, 3) << endl;
    function<float(float)> div_by_6 = std::bind(divide, _1, 6);
    cout << "1/6: div_by_6(1) = " << div_by_6(1) << endl;
    cout << "2/6: div_by_6(2) = " << div_by_6(2) << endl;
    cout << "3/6: div_by_6(3) = " << div_by_6(3) << endl;

    // New: function object from a member function using bind or mem_fn
    cout << "\nFunction object on member functions:-------------------------------\n";
    Vector3                   vecn   = normalize({.5f, .5f, .5f}); // Passing a Vector3 to normalize
    function<float(Vector3&)> length = std::mem_fn(&Vector3::length);
    cout << "length(vn) = " << length(vecn) << endl;

    cout << "\nFunction object on member functions using bind:--------------------\n";
    function<float(void)> vecLength = std::bind(&Vector3::length, vecn);
    cout << "vecLength() = " << vecLength() << endl;
}
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Not thread save counter to demonstrate thread interleaving
struct CounterNTS
{
    int  value;
    void inc() { ++value; }
    void dec() { --value; }
};
//-----------------------------------------------------------------------------
// Thread safe counter with mutex
struct CounterTS1
{
    std::mutex mx;
    int        value = 0;
    void       inc()
    {
        mx.lock();
        ++value;
        mx.unlock();
    }
    void dec()
    {
        mx.lock();
        --value;
        mx.unlock();
    }
};
//-----------------------------------------------------------------------------
// Thread safe counter with atomic in value
struct CounterTS2
{
    std::atomic<int> value = 0;
    void             inc() { ++value; }
    void             dec() { --value; }
};
//-----------------------------------------------------------------------------
// some global objects and functions called from threads
void call_from_thread(int tid) { cout << "Launched by thread " << tid << endl; }

CounterNTS cntNTS;
void       incNTS()
{
    for (int i = 0; i < 10000; ++i) cntNTS.inc();
}
void decNTS()
{
    for (int i = 0; i < 10000; ++i) cntNTS.dec();
}

CounterTS1 cntTS1;
void       incTS1()
{
    for (int i = 0; i < 10000; ++i) cntTS1.inc();
}
void decTS1()
{
    for (int i = 0; i < 10000; ++i) cntTS1.dec();
}

CounterTS2 cntTS2;
void       incTS2()
{
    for (int i = 0; i < 10000; ++i) cntTS2.inc();
}
void decTS2()
{
    for (int i = 0; i < 10000; ++i) cntTS2.dec();
}

std::mutex myMutex;
//-----------------------------------------------------------------------------
// This function will be called from a thread. The output stream is protected
void call_from_thread2(int tid)
{
    // Lock & unlock yourself
    // myMutex.lock();
    // cout << "Launched by thread " << tid << endl;
    // myMutex.unlock();

    // Only lock & auto unlock at block exit
    std::lock_guard<std::mutex> guard(myMutex);
    cout << "Launched by thread " << tid << endl;
}

///////////////////////////////////////////////////////////////////////////////
void new_threading()
{
    high_resolution_clock::time_point t1, t2;

    cout << "\nNew: Threads: -----------------------------------------------------\n";
    // It is important two know how many real concurrent threads can run on a
    // system. A quad core with hyper thread returns here 8.
    cout << "thread::hardware_concurrency: " << thread::hardware_concurrency() << endl;

    cout << "\nNew: Threads: cout without mutex ----------------------------------\n";
    const int NUM_THREADS = 10;

    // Declare a group of threads
    thread t[NUM_THREADS];

    // Launch a group of threads that start with call_from_thread1 with int argument
    for (int i = 0; i < NUM_THREADS; ++i)
        t[i] = thread(call_from_thread, i);
    cout << "Launched from the main\n";

    // Join the threads will cause the main thread to wait for all threads
    for (int i = 0; i < NUM_THREADS; ++i) t[i].join();

    cout << "\nNew: Threads: cout with mutex -------------------------------------\n";
    // Launch a group of threads that start with call_from_thread2 with int argument
    for (int i = 0; i < NUM_THREADS; ++i)
        t[i] = thread(call_from_thread2, i);

    myMutex.lock();
    cout << "Launched from the main\n";
    myMutex.unlock();

    // Join the threads with the main thread
    for (int i = 0; i < NUM_THREADS; ++i) t[i].join();

    cout << "\nNew: Threads: counter without mutex -------------------------------\n";
    // Let's compare the sync penalty of the mutex protected CounterTS1.
    // First run the not thread save version. The test runs 10 times and the threads
    // increment 1000 times and decrement 1000 times. So the final result on the
    // counter should be zero:

    t1 = high_resolution_clock::now();
    for (int j = 0; j < 10; j++)
    {
        vector<thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            threads.push_back(thread(incNTS));
            threads.push_back(thread(decNTS));
        }
        for (auto& thread : threads) { thread.join(); }
        cout << cntNTS.value << ",";
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    cout << "\nNew: Threads: counter with mutex ----------------------------------\n";
    // Now we do the same test with thread safe counter that protects the value
    // with a mutex:

    t1 = high_resolution_clock::now();
    for (int j = 0; j < 10; j++)
    {
        vector<thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            threads.push_back(thread(incTS1));
            threads.push_back(thread(decTS1));
        }
        for (auto& thread : threads) { thread.join(); }
        cout << cntTS1.value << ",";
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    cout << "\nNew: thread: counter with atomic int ------------------------------\n";
    // The last test protects the counter value with an atomic int value without
    // mutex. This should be a lot faster:

    t1 = high_resolution_clock::now();
    for (int j = 0; j < 10; j++)
    {
        vector<thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            threads.push_back(thread(incTS2));
            threads.push_back(thread(decTS2));
        }
        for (auto& thread : threads) { thread.join(); }
        cout << cntTS2.value.load() << ",";
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    cout << "\nNew: async: Simple concurrency ------------------------------------\n";
    // For simple heavy tasks that don't need any synchronization we can launch an
    // asynchronous function with async and wait for the result with a future variable.

    std::future<int> futureResult(std::async([](int m, int n)
                                             { return m + n; },
                                             2,
                                             4));
    cout << "Do something complex here ..." << endl;
    cout << "And get the the future result afterwards: " << futureResult.get() << endl;
}
//////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_random_generators()
{
    std::default_random_engine         generator;
    std::uniform_int_distribution<int> distributionInt(1, 6);
    auto                               diceRoll = bind(distributionInt, generator);
    cout << "\nRandom dice rolls:-------------------------------------------------\n";
    for (int i = 0; i < 20; ++i) cout << distributionInt(generator) << ",";
    cout << endl;
    for (int i = 0; i < 20; ++i) cout << diceRoll() << ",";
    cout << endl;

    std::mt19937                           mersenneTwister;
    std::uniform_real_distribution<double> distributionDouble(0, 1);
    auto                                   rnd = bind(distributionDouble, mersenneTwister);
    cout << "\nRandom numbers with mersenne twister:------------------------------\n";
    for (int i = 0; i < 5; ++i) cout << distributionDouble(mersenneTwister) << ",";
    cout << endl;
    for (int i = 0; i < 5; ++i) cout << rnd() << ",";
    cout << endl;
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_smart_pointers()
{
    high_resolution_clock::time_point  t1, t2;
    std::default_random_engine         generator;
    std::uniform_int_distribution<int> distributionInt(1, 6);
    auto                               diceRoll = bind(distributionInt, generator);

    cout << "\nNew: unique pointers:---------------------------------------------\n";
    // use unique_ptr for single ownership dynamic allocated objects.
    // No delete is needed because object is deleted automatically at scope end.
    // Old style:
    {
        A* pA = new A();
        pA->f1();
    }
    cout << "oops, memory leak" << endl;

    // New style:
    {
        unique_ptr<A> upA1(new A);
        unique_ptr<A> upA2 = std::make_unique<A>(); // C++14

        upA1->f1();
        upA2->f1();

        // A unique pointer is unique. It can not be assigned to another unique pointer
        // unique_ptr<A> upA3 = upA1; // compile error
    }
    cout << "No, memory leak." << endl;

    // if you need multiple instancies of dynamic objects use a std::vector or std::array.
    // Don't attach a C-array to a unique_ptr. It doesn't have the [] operator.
    // Old style:
    {
        int num = diceRoll();
        A*  pA  = new A[num];
        for (int i = 0; i < num; i++)
            pA[i].f1();
    }
    cout << "oops, memory leak" << endl;

    // New style:
    {
        vector<unique_ptr<A>> vupA;
        for (int i = 0; i < diceRoll(); i++)
            vupA.push_back(unique_ptr<A>(new A()));
        for (size_t i = 0; i < vupA.size(); i++)
            vupA[i]->f1();
    }
    cout << "No, memory leak." << endl;

    cout << "\nNew: shared pointer:---------------------------------------------\n";
    // If you need more than one pointer to share the same object use shared_ptr.
    // The use reference counting to determine the object deallocation.
    // With shared ownership no one is responsible for the object deallocation.

    // Stroustroup recommends to be careful with shared_ptr. You should prefer
    // unique_ptr when possible. Read http://stroustrup.com/C++11FAQ.html#std-shared_ptr

    // Shared_ptr are also more costly than unique_ptr because they allocate an
    // additional managing object. So always 2 allocation are done for one shared_ptr.

    {
        // shared_ptr<A> pA1(new A());   // count is 1
        shared_ptr<A> pA1 = std::make_shared<A>(); // alternative without new C++11 make_shared
        // shared_ptr<A> pA1 = new A(); // Error no assignement of a raw pointer allowed

        pA1->f1();
        cout << "Count after 1st shared pointer: " << pA1.use_count() << endl;
        {
            shared_ptr<A> pA2(pA1); // count is 2
            pA2->f1();
            cout << "Count after 2nd shared pointer: " << pA1.use_count() << endl;
            {
                shared_ptr<A> pA3 = pA1; // creation by assignement, count is 3
                pA3->f1();
                cout << "Count after 3rd shared pointer: " << pA1.use_count() << endl;
            } // count is 2
            cout << "Count after 3rd share pointer is gone: " << pA1.use_count() << endl;
        }     // count is 1
        cout << "Count after 2nd share pointer is gone: " << pA1.use_count() << endl;
    }
    cout << "No, memory leak." << endl;

    //--------------------------------------------------------
    // Performance with new/delete
    static const unsigned int numInt = 10000000;
    t1                               = high_resolution_clock::now();
    for (unsigned int i = 0; i < numInt; ++i)
    {
        int* tmp(new int(i));
        delete tmp;
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used for new/delete    : " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // Performace with unique pointers
    t1 = high_resolution_clock::now();
    for (long long i = 0; i < numInt; ++i)
    {
        unique_ptr<int> tmp(std::make_unique<int>(i));
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used for unique pointer: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // Performace with shared pointers
    t1 = high_resolution_clock::now();
    for (long long i = 0; i < numInt; ++i)
    {
        shared_ptr<int> tmp(std::make_shared<int>(i));
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime used for shared pointer: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // If you need more than one pointer to point to the same object but only
    // the shared_ptr is responsible for the deallocation use for the other
    // pointers a weak_ptr. Weak pointers do not increment or decrement the
    // shared reference counter. Weak pointer can be used to solve the problem
    // of cyclic shared_ptr.
    // See http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf
}
///////////////////////////////////////////////////////////////////////////////

constexpr uint64_t fibonacci(uint64_t n) { return n <= 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2); }
constexpr uint64_t factorial(uint64_t n) { return n <= 1 ? 1 : (n * factorial(n - 1)); }

///////////////////////////////////////////////////////////////////////////////
void new_const_expression()
{
    // New C++11: Constant expression evaluated at compile time
    // They allow like template meta programming the
    // evaluation and execution at compile time.
    // New C++17: constexpr is allowed in if statments

    cout << "\nNew C++11: constexpr evaluated at compile time:--------------------\n";
    cout << "fibonacci(10) = " << fibonacci(20) << endl;
    cout << "factorial(10) = " << factorial(20) << endl;
}
///////////////////////////////////////////////////////////////////////////////

// Simple class that holds a length in meters.
class Length
{
public:
    class Meter
    {
    }; // a tag class for avoiding implicit constructor call with double
    explicit constexpr Length(Meter, double lenInM) : _lengthInMeters(lenInM) {}

    // New Cpp17 attribute nodiscard leads to a warning if a caller ignores the return value
    [[nodiscard]] double lenghtInMeters() const { return _lengthInMeters; }

    Length operator+(const Length& l) const { return Length(Meter{}, l.lenghtInMeters() + _lengthInMeters); }

private:
    double _lengthInMeters;
};

// User defined literals must begin with a _ for not clashing with futer std::literals
// constexpr is optional. If applied it is evaluated at compile time.
constexpr Length operator"" _m(long double len) { return Length(Length::Meter(), (double)(len)); }
constexpr Length operator"" _mm(long double len) { return Length(Length::Meter(), (double)(len * 0.001)); }
constexpr Length operator"" _cm(long double len) { return Length(Length::Meter(), (double)(len * 0.01)); }
constexpr Length operator"" _dm(long double len) { return Length(Length::Meter(), (double)(len * 0.1)); }
constexpr Length operator"" _km(long double len) { return Length(Length::Meter(), (double)(len * 1000)); }
constexpr Length operator"" _in(long double len) { return Length(Length::Meter(), (double)(len * 0.0254)); }
constexpr Length operator"" _ft(long double len) { return Length(Length::Meter(), (double)(len * 0.3048)); }
constexpr Length operator"" _yd(long double len) { return Length(Length::Meter(), (double)(len * 0.9144)); }
constexpr Length operator"" _ml(long double len) { return Length(Length::Meter(), (double)(len * 1609.34)); }

///////////////////////////////////////////////////////////////////////////////
void new_userdefined_literals()
{

    cout << "\nNew C++11: User defined literals:----------------------------------\n";
    // See the definition of class Length in CPP1.h
    // See https://www.codeproject.com/Articles/447922/Application-of-Cplusplus11-User-Defined-Literals-t
    // for a more sophisticated unit system
    // We have an f as literal for floats
    //           \/
    float f = 1.0f;

    // Length len0 = 1.0;    // Compile Error
    Length len1 = 1.0_m;
    cout << len1.lenghtInMeters() << " m" << endl;
    Length len2 = 2.0_in + 2.0_km;
    cout << len2.lenghtInMeters() << " m" << endl;

    cout << "\nNew C++14: New std::chrono_literals:-------------------------------\n";
    using namespace std::chrono_literals;
    auto                      d1 = 250ms;
    std::chrono::milliseconds d2 = 1s;
    std::chrono::seconds      d3 = 1min;
    std::cout << "250ms = " << d1.count() << " milliseconds\n"
              << "   1s = " << d2.count() << " milliseconds\n"
              << " 1min = " << d3.count() << " seconds\n";
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
void new_if_switch_statement()
{
    cout << "\nNew C++17: if-switch-statement:------------------------------------\n";

    const string myString = "Hello World";

    auto it1 = myString.find("Hello");
    if (it1 != string::npos)
        cout << it1 << " Hello\n";

    // C++17 with init if: Great, we saved a line ;-)
    if (const auto it2 = myString.find("Hello"); it2 != string::npos)
        cout << it2 << " Hello\n";
}
///////////////////////////////////////////////////////////////////////////////

#include <set>
#include <tuple> // for tie

struct S
{
    int    n;
    string s;
    float  d;
    bool   operator<(const S& rhs) const
    {
        // compares n to rhs.n,
        //     then s to rhs.s,
        //     then d to rhs.d
        return tie(n, s, d) < tie(rhs.n, rhs.s, rhs.d);
    }
};

///////////////////////////////////////////////////////////////////////////////
void new_structured_binding()
{
    cout << "\nNew C++17: structured binding:-------------------------------------\n";

    std::set<S> mySet;

    // pre C++17:
    {
        S                     value{42, "Test", 3.14f};
        std::set<S>::iterator iter;
        bool                  inserted = false;

        // unpacks the return val of insert into iter and inserted
        tie(iter, inserted) = mySet.insert(value);

        if (inserted)
            cout << "Value(" << iter->n << ", " << iter->s << ", ...) was inserted"
                 << "\n";
    }

    // with C++17:
    {
        S value{100, "abc", 100.0};
        const auto [iter, inserted] = mySet.insert(value);

        if (inserted)
            cout << "Value(" << iter->n << ", " << iter->s << ", ...) was inserted"
                 << "\n";
    }
}
///////////////////////////////////////////////////////////////////////////////

#ifndef __APPLE__
#    include <filesystem>
namespace fs = std::filesystem;

uintmax_t computeFileSize(const fs::path& file)
{
    if (fs::exists(file) && fs::is_regular_file(file))
    {
        auto err      = std::error_code{};
        auto filesize = fs::file_size(file, err);
        if (filesize != static_cast<uintmax_t>(-1))
            return filesize;
    }

    return 0;
}

void displayDirectoryTreeImp(const fs::path& pathToShow, int level)
{
    if (fs::exists(pathToShow) && fs::is_directory(pathToShow))
    {
        auto lead = string(level * 4, ' ');
        for (const auto& entry : fs::directory_iterator(pathToShow))
        {
            auto filename = entry.path().filename();
            if (fs::is_directory(entry.status()))
            {
                cout << lead << "[+] " << filename << "\n";
                displayDirectoryTreeImp(entry, level + 1);
                cout << "\n";
            }
            else if (fs::is_regular_file(entry.status()))
                cout << lead << " " << filename << ", " << computeFileSize(entry) << "\n";
            else
                cout << lead << "  [?]" << filename << "\n";
        }
    }
}

void displayDirectoryTree(const fs::path& pathToShow)
{
    displayDirectoryTreeImp(pathToShow, 0);
}

///////////////////////////////////////////////////////////////////////////////
void new_filesystem()
{
    cout << "\nNew C++17: filesystem access:--------------------------------------\n";

    fs::path myFile("z:/hudrima1/Dropbox/BFH/Module/7503-Adv-C++/Exercises/11_CPP1z/CPP1z.cpp");

    cout << "exists()          = " << fs::exists(myFile) << "\n"
         << "is_regular_file() = " << fs::is_regular_file(myFile) << "\n"
         << "is_directory()    = " << fs::is_directory(myFile) << "\n"
         << "file_size()       = " << computeFileSize(myFile) << "\n"
         << "root_name()       = " << myFile.root_name() << "\n"
         << "root_path()       = " << myFile.root_path() << "\n"
         << "relative_path()   = " << myFile.relative_path() << "\n"
         << "parent_path()     = " << myFile.parent_path() << "\n"
         << "filename()        = " << myFile.filename() << "\n"
         << "stem()            = " << myFile.stem() << "\n"
         << "extension()       = " << myFile.extension() << "\n\n";

    // Get parts of a path
    cout << "\nNFilesystem path parts:---------------------------------------------\n";
    int i = 0;
    for (const auto& part : myFile)
        cout << "path part: " << i++ << " = " << part << "\n";

    // Print a directory tree
    cout << "\nFilesystem recursive traversal:--------------------------------------\n";
    fs::path myDir("z:/hudrima1/Dropbox/BFH/Module/7503-Adv-C++/Exercises/11_CPP1z");
    displayDirectoryTree(myDir);
}
///////////////////////////////////////////////////////////////////////////////
#endif

#ifndef __APPLE__
///////////////////////////////////////////////////////////////////////////////
void new_parallel_algorithms()
{
    high_resolution_clock::time_point t1, t2;

    cout << "\nNew C++17: parallel algorithms:------------------------------------\n";
    vector<int> r;
    for (int i = 0; i < 1000000; ++i)
        r.push_back(rand());

    // standard sequential sort
    t1 = high_resolution_clock::now();
    sort(r.begin(), r.end());
    t2 = high_resolution_clock::now();
    cout << "Standard sort(old)                      :" << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // explicitly sequential sort
    t1 = high_resolution_clock::now();
    sort(std::execution::seq, r.begin(), r.end());
    t2 = high_resolution_clock::now();
    cout << "Standard sort(std::execution::seq)      :" << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // permitting parallel execution
    t1 = high_resolution_clock::now();
    sort(std::execution::par, r.begin(), r.end());
    t2 = high_resolution_clock::now();
    cout << "Standard sort(std::execution::par)      :" << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // permitting vectorization as well
    t1 = high_resolution_clock::now();
    sort(std::execution::par_unseq, r.begin(), r.end());
    t2 = high_resolution_clock::now();
    cout << "Standard sort(std::execution::par_unseq):" << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;
}
///////////////////////////////////////////////////////////////////////////////
#endif