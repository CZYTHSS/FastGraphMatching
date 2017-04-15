#include <iostream>
using namespace std;
#include <omp.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <string.h>

int main(){
	ifstream fin("emd_test");
	char* line = new char[100000000];
	fin.getline(line,100000000);
	cout << line << endl;
	while(!fin.eof() && strlen(line) == 0){
		fin.getline(line, 100000000);
		cout << line << endl;
	}

	int a,b;
	sscanf(line,"%d%d",&a, &b);
	//sscanf(line, "%d", &b);
	cout << a << "*" << b << " matrix."<<endl;

	return 0;
}
