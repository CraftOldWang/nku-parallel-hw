#include <iostream>

using namespace std;

// extern void test_nine_func();
// extern void correct_test();
extern void correct_test_avx();

int main(){

    
    // test_nine_func();
    // correct_test();
    correct_test_avx();

    cout<<"ALL TEST PASSED"<<endl;

    return 0;
}