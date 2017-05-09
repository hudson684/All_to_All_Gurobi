#include <iostream>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <omp.h>
#include "gurobi_c++.h"


using namespace std;

#define nComp 6
#define nFiles 6

#define TwoToTheNFiles 512
#define TwoToTheNComps 512


#define lowerBound 2
#define upperBound 4

#define maxDiff 1

#define maxNum 72281345626

int count_so_far = 0;
int currPercent = 0;

double best = 1;
int BoolValArray[TwoToTheNComps];

void populateArray() {

	BoolValArray[0] = 0;
	BoolValArray[1] = 1;
	int N2 = 1;

	for (int i = 2; i < TwoToTheNComps; i++)
	{
		//cout << "i :" << i << endl;
		if ((i & (i - 1)) == 0) {
			N2 = i;
			//cout << endl;
		}
		BoolValArray[i] = 1 + BoolValArray[i - N2];

		/*cout << " i: ";

		//if (i < 10) {
		//cout << "0";
		//}
		//cout << i << " val: " << BoolValArray[i] << endl;*/
	}
}


int checkArray(int Index) {
	return BoolValArray[Index];
}



double getPercentage(int ints[nFiles]) {
	count_so_far++;

	GRBEnv env = GRBEnv();

	GRBModel model = GRBModel(env);
	model.getEnv().set(GRB_IntParam_LogToConsole, 0);
	

	// Create variables
	GRBVar vars[nFiles + 2];
	for (int v = 0; v < nFiles + 2; v++) {
		string s = "V" + to_string(v);
		vars[v] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, s);
	}

	model.update();
	GRBVar X = vars[nFiles];
	GRBVar Y = vars[nFiles + 1];
	GRBLinExpr Lr1 = (X + Y);



	int comps[nComp];
	GRBLinExpr linExprs[nComp + 1];
	for (int k = 0; k < nComp; k++)
	{
		comps[k] = 0;
		linExprs[k] = 0;
	}


	int numDots = 0;
	for (int j = 0; j < nFiles; j++)
	{
		//cout << " int: ";
		if (ints[j] < 10) {
			//cout << 0;
		}

		if (ints[j] < 100) {
			//cout << 0;
		}

		//cout << ints[j] << " ";

		for (int i = 0; i < nComp; i++)
		{
			int power = pow(2, i);
			if ((ints[j] & power) != 0) {
				//cout << " x";
				numDots++;
				linExprs[i] += vars[j];
			}
			else {
				//cout << " o";
			}

			

		}
		


		//cout << endl;


	}

	

	//cout << endl;

	model.setObjective(Lr1, GRB_MINIMIZE);
	for (int p = 0; p < nFiles + 1; p++)
	{
		if (p < nComp) {
			linExprs[nComp] += vars[p];
			string c = "C" + to_string(p);
			model.addConstr(linExprs[p] - Lr1 <= 0, c);
		}
		//cout << "Lin Expresions : " << p << ": " << linExprs[p] << " " << endl;

	}

	model.addConstr(linExprs[nComp] == 1, "c_Final");

	

	//cout << endl;
	model.optimize();
	

	for (int i = 0; i < nFiles + 2; i++) {
		//cout << vars[i].get(GRB_StringAttr_VarName) << " " << vars[i].get(GRB_DoubleAttr_X) << endl;
	}

	//cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

	

	//cout << endl;
	//cout << endl;






	return model.get(GRB_DoubleAttr_ObjVal);
	//return 1;
}

void searchSpace(int curXDepth, int curYDepth, int maxXDepth, int maxYDepth, int ints[nFiles]) {

	if (curYDepth == 0)
	{
		#pragma omp parallel for schedule(dynamic, 1)
		for (int i = curXDepth + 1; i <= maxXDepth - (maxYDepth - curYDepth); i++)
		{

			#pragma omp critical 
			{
				int id = omp_get_thread_num();
				std::cout << " Thread Num: " << id << "i: " << i << endl;
			}
			if (checkArray(i) > lowerBound && checkArray(i) < upperBound) {
				ints[curYDepth] = i;
				searchSpace(i, curYDepth + 1, maxXDepth, maxYDepth, ints);
			}
		}

	}
	else {

		for (int i = curXDepth + 1; i <= maxXDepth - (maxYDepth - curYDepth); i++)
		{
			if (checkArray(i) > lowerBound && checkArray(i) < upperBound) {

				bool pass = true;

				//if the current number of files is equal to or greater to the upperBound,
				//then there is a chance for a computer to have that many files
				//this will rule out computers that have too many files.
				//for each computer, find how many files are on it, if files > upperBound
				//pass = false
				if (curYDepth >= upperBound && pass) {
					for (int j = 0; j < nComp; j++) {
						int tally = 0;
						for (int k = 0; k < curYDepth; k++) {
							int cur_comp = pow(2.0, j);
							int l = (ints[k] & cur_comp);
							if (k != 0) {
								tally++;
							}
						}
						if (tally >= upperBound) {
							pass = false;
						}
					}
				}

				for (int j = 0; j < curYDepth; j++)
				{
					if (pass) {

						if (checkArray(ints[j]) - checkArray(i) > maxDiff)
						{
							pass = false;
						}

						int k = (ints[j] & i);
						if (k == 0) {

							pass = false;
						}

					}
				}

				if (pass)
				{


					if (curYDepth < maxYDepth)
					{
						ints[curYDepth] = i;
						searchSpace(i, curYDepth + 1, maxXDepth, maxYDepth, ints);

					}
					else
					{
						ints[maxYDepth] = i;
						double current_best = getPercentage(ints);
						if (current_best < best) {
							best = current_best;
						}

					}
				}
			}
		}
	}
}

const int total = 128;
const int depth = 7;
int counter_count = 0;

void counter(int x, int y) {
	
	for (int i = x + 1; i <= (total - (depth + y + 1)); i++) {
		if (y == 0) {
			cout << "I++" << endl;
		}

		if (y < depth) {
			counter(i, y + 1);
		}
		else {
			counter_count++;
		}
	}

}

void output()
{

	cout << "number of computers: " << nComp << endl;
	cout << "number of files: " << nFiles << endl;

	cout << "Best is: " << best << endl;

}




int main(int argc, char *argv[]){

	clock_t start;
	double duration;

	populateArray();


	start = clock();

	int ints[nFiles];
	for (int i = 0; i < nFiles; i++)
	{
		ints[i] = 0;
	}

	int xMax = pow(2.0, nComp);
	cout << "maximum i: " << xMax << endl;

	searchSpace(0, 0, xMax - 1, nFiles - 1, ints);
	output();
	

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "time taken: " << duration << '\n';



	return 0;
}