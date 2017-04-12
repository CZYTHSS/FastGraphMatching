#include <iostream>
using namespace std;
#include <omp.h>
#include <stdio.h>

int main(){
	for(int i = 0; i < 10000; i++){
		cout << i << endl;
	}
	return 0;
}
