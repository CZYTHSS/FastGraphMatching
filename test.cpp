#include <iostream>
using namespace std;
#include <omp.h>
#include <stdio.h>
#include <fstream>

int main(){
	ofstream fout;
	fout.open("result");
	fout << "hello world\n";
	int i = 0;
	fout << i << " " << i + 1 << endl;
	fout.close();
	
	return 0;
}
