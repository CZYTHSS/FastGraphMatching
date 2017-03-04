#include "problem.h"
#include "factor.h"
#include <time.h>

double prediction_time = 0.0;
extern Stats* stats;
bool debug = false;

void exit_with_help(){
	cerr << "Usage: ./predict (options) [testfile] [model]" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Viterbi(chain)" << endl;
	cerr << "	1 -- sparseLP" << endl;
	cerr << "       2 -- GDMM" << endl;
	cerr << "-p problem_type: " << endl;
	cerr << "   chain -- sequence labeling problem" << endl;
	cerr << "   network -- network matching problem" << endl;
	cerr << "   uai -- uai format problem" << endl;
	cerr << "-e eta: GDMM step size" << endl;
	cerr << "-o rho: coefficient/weight of message" << endl;
	cerr << "-m max_iter: max number of iterations" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	vector<string> args;
	for (i = 1; i < argc; i++){
		string arg(argv[i]);
		//cerr << "arg[i]:" << arg << "|" << endl;
		args.push_back(arg);
	}
	for(i=0;i<args.size();i++){
		string arg = args[i];
		if (arg == "-debug"){
			debug = true;
			continue;
		}
		if( arg[0] != '-' )
			break;

		if( ++i >= args.size() )
			exit_with_help();

		string arg2 = args[i];

		if (arg == "--printmodel"){
			param->print_to_loguai2 = true;
			param->loguai2fname = arg2;
			continue;
		}
		switch(arg[1]){
			case 's': param->solver = stoi(arg2);
				  break;
			case 'e': param->eta = stof(arg2);
				  break;
			case 'o': param->rho = stof(arg2);
				  break;
			case 'm': param->max_iter = stoi(arg2);
				  break;
			case 'p': param->problem_type = string(arg2);
				  break;
			default:
				  cerr << "unknown option: " << arg << " " << arg2 << endl;
				  exit(0);
		}
	}

	if(i>=args.size())
		exit_with_help();

	param->testFname = argv[i+1];
	i++;
	if( i<args.size() )
		param->modelFname = argv[i+1];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

double struct_predict(Problem* prob, Param* param){
	Float hit = 0.0;
	Float N = 0.0;
	int n = 0;
	stats = new Stats();
    int K = prob->K;
    vector<UniFactor*> x;
    for (int i = 0; i < K; i++){
        UniFactor* x_i = new UniFactor(K, prob->node_score_vecs[i], param);
        x.push_back(x_i);
    }
    vector<UniFactor*> xt;
    for (int i = 0; i < K; i++){
        UniFactor* xt_i = new UniFactor(K, prob->node_score_vecs[K+i], param);
        xt.push_back(xt_i);
    }
    int iter = 0;
    Float rho = param->rho;
    int* indices = new int[K];
    for (int i = 0; i < K*2; i++){
        indices[i] = i;
    }
    while (iter++ < param->max_iter){
        random_shuffle(indices, indices+K*2);
        for (int k = 0; k < K*2; k++){
            if (indices[k] < K){
                int i = indices[k];
                UniFactor* node = x[i];
                node->subsolve();
                Float* msg = node->msg;
                Float* y = node->y;
                for (int j = 0; j < K; j++){
                    xt[j]->msg[i] = -msg[j];
                }
            } else {
                int j = indices[k]-K;
                UniFactor* node = xt[j];
                node->subsolve();
                Float* msg = node->msg;
                Float* y = node->y;
                for (int i = 0; i < K; i++){
                    x[i]->msg[j] = -msg[i];
                    //cout << "msg_t=" << msg[i] << ", xt[j][i]=" << y[i] << ", x[i][j]=" << x[i]->y[j] << ", msg=" << x[i]->msg[j] << endl;
                }
            }
        }
        // msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
        // msg[i] = (x[i][j] - xt[j][i] + mu[i][j])
        for (int i = 0; i < K; i++){
            for (int j = 0; j < K; j++){
                Float delta = x[i]->y[j] - xt[j]->y[i];
                x[i]->msg[j] += delta;
                xt[j]->msg[i] -= delta;
            }
        }
        for (int i = 0; i < K; i++){
            for (int j = 0; j < K; j++){
                cout << x[i]->y[j]*x[i]->c[j] << "\t";
            }
            cout << "\t";
            for (int j = 0; j < K; j++){
                cout << xt[j]->y[i]*xt[j]->c[i] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

	cerr << "uni_search=" << stats->uni_search_time
		<< ", uni_subsolve=" << stats->uni_subsolve_time
		<< ", bi_search=" << stats->bi_search_time
		<< ", bi_subsolve=" << stats->bi_subsolve_time 
		<< ", maintain=" << stats->maintain_time 
		<< ", construct=" << stats->construct_time << endl; 
	return (double)hit/(double)N;
}


int main(int argc, char** argv){
	if (argc < 2){
		exit_with_help();
	}

	prediction_time = -get_current_time();
	srand(time(NULL));
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);

	Problem* prob = NULL;
    if (param->problem_type=="bipartite"){
        prob = new BipartiteMatchingProblem(param);
        prob->construct_data();
        int K = ((BipartiteMatchingProblem*)prob)->K;
        cerr << "prob.K=" << K << endl;
    }

	if (prob == NULL){
		cerr << "Need to specific problem type!" << endl;
	}

	cerr << "param.rho=" << param->rho << endl;
	cerr << "param.eta=" << param->eta << endl;

	/*
	   double t1 = get_current_time();
	   vector<Float*> cc;
	   for (int i = 0; i < 200; i++){
	   Float* temp_float = new Float[4];
	   cc.push_back(temp_float);
	   }
	   for (int tt = 0; tt < 3000*1000; tt++)
	   for (int i = 0; i < 200; i++){
	   Float* cl = cc[rand()%200];
	   Float* cr = cc[rand()%200];
	   for (int j = 0; j < 4; j++)
	   cl[j] = cr[j];
	   }
	   cerr << get_current_time() - t1 << endl;
	 */

    if (param->solver == 2){
		cerr << "Acc=" << struct_predict(prob, param) << endl;
	}
	prediction_time += get_current_time();
	cerr << "prediction time=" << prediction_time << endl;
	return 0;
}
