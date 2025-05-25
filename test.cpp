//
// Created by CraftOldW on 25-5-25.
//
#include <iostream>
#include <vector>
using namespace  std;

int main()
{
  vector<string> test;
  string a= "testa";
  string b = "hello";
  test.emplace_back(a+b);
  cout << test[0]<<endl;

  return 0;
}