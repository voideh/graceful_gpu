#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
int main()
{
    string s = "0,1,12,13,4";
    do{
        cout << s << endl;
    }while(next_permutation(s.begin(), s.end()));
    return 0xc0ff3;
}
