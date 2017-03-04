#ifndef FACTOR_H
#define FACTOR_H

#include "util.h"
#include "stats.h"

Stats* stats = new Stats();

class Factor{
	public:
		virtual inline void search(){
		}
		virtual inline void subsolve(){
		}
};


//unigram factor, y follows simplex constraints
class UniFactor : public Factor{
	public:
		//fixed
		int K;
		Float rho;
		Float* c; // score vector, c[k] = -<w_k, x>
		Float nnz_tol;

		bool shrink;

		//maintained
		Float* grad;
		Float* y;
        bool* inside;
		Float* msg;
		vector<int> act_set;
		//vector<int> ever_act_set;
		//bool* is_ever_act;
		int searched_index;

		inline UniFactor(int _K, Float* _c, Param* param){
			K = _K;
			rho = param->rho;
			nnz_tol = param->nnz_tol;
			//compute score vector
			c = _c;
			//cache of gradient
			grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);

			//relaxed prediction vector
			y = new Float[K];
			memset(y, 0.0, sizeof(Float)*K);

			inside = new bool[K];
			memset(inside, false, sizeof(bool)*K);
            msg = new Float[K];
			memset(msg, 0.0, sizeof(Float)*K);
			act_set.clear();

            fill_act_set();
			shrink = false;
		}

		~UniFactor(){
			delete[] y;
			delete[] grad;
			delete[] inside;
			//delete[] is_ever_act;
			act_set.clear();
			delete msg;
		}

		void fill_act_set(){
			act_set.clear();
			//ever_act_set.clear();
			for (int k = 0; k < K; k++){
				act_set.push_back(k);
				//ever_act_set.push_back(k);
				//is_ever_act[k] = true;
				inside[k] = true;
			}
		}

		//uni_search()
		//inline void search(){
		//	stats->uni_search_time -= get_current_time();
		//	//compute gradient of y_i
		//	for (int k = 0; k < K; k++){
		//		grad[k] = c[k];
		//	}
		//	//for (vector<Float*>::iterator m = msgs.begin(); m != msgs.end(); m++){
		//	//	Float* msg = *m;
		//	//	for (int k = 0; k < K; k++)
		//	//		grad[k] += rho * msg[k];
		//	//}
		//	Float gmax = -1e100;
		//	int max_index = -1;
		//	for (int k = 0; k < K; k++){
		//		if (inside[k]) continue;
		//		//if not inside, y_k is guaranteed to be zero, and y_k is nonnegative, so we only care gradient < 0
		//		//if (grad[k] > 0 && act_set.size() != 0) continue;
		//		if (-grad[k] > gmax){
		//			gmax = -grad[k];
		//			max_index = k;
		//		}
		//	}

		//	searched_index = max_index;
		//	if (max_index != -1){
		//		act_set.push_back(max_index);
		//		inside[max_index] = true;
		//	}
		//	stats->uni_search_time += get_current_time();
		//}


		//	min_{y \in simplex} <c, y> + \rho/2 \sum_{msg \in msgs} \| (msg + y) - y \|_2^2
		// <===>min_{y \in simplex} \| y - 1/|msgs| ( \sum_{msg \in msgs} (msg + y) - 1/\rho c ) \|_2^2
		// <===>min_{y \in simplex} \| y - b \|_2^2
		// uni_subsolve()
		//bool output = false;
		inline void subsolve(){
			if (act_set.size() == 0)
				return;
			stats->uni_subsolve_time -= get_current_time();
			Float* y_new = new Float[act_set.size()];
			int act_count = 0;

            Float* b = new Float[act_set.size()];
            memset(b, 0.0, sizeof(Float)*act_set.size());
            act_count = 0;
            for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
                int k = *it;
                b[act_count] -= c[k]/rho;
                b[act_count] -= msg[k]-y[k];
                //cerr << b[act_count] << " ";
            }
            cout << "solving simplex:" << endl;
            cout << "\t";
            for (int k = 0; k < act_set.size(); k++){
                cout << " " << b[k];
            }
            cout << endl;
            //cerr << endl;
            solve_simplex(act_set.size(), y_new, b);
            cout << "\t";
            for (int k = 0; k < act_set.size(); k++){
                cout << " " << y_new[k];
            }
            cout << endl;
            delete[] b;
            
            act_count = 0;
            for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
                int k = *it;
                Float delta_y = y_new[act_count] - y[k];
                //stats->delta_Y_l1 += fabs(delta_y);
                msg[k] += delta_y;
            }

			vector<int> next_act_set;
			next_act_set.clear();
			act_count = 0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++, act_count++){
				int k = *it;
				y[k] = y_new[act_count];
				//shrink
				if (!shrink || y[k] >= nnz_tol ){
					//this index is justed added by search()
					//Only if this index is active after subsolve, then it's added to ever_act_set
					/*if (k == searched_index){
						adding_ever_act(k);
					}*/
					next_act_set.push_back(k);
				} else {
					inside[k] = false;
				}
			}
			act_set = next_act_set;

			delete[] y_new;
			stats->uni_subsolve_time += get_current_time();
		}

		int recent_pred = -1;
		//goal: minimize score
		inline Float score(){
			/*Float score = 0.0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				score += c[k]*y[k];
			}
			return score;
			*/
			Float max_y = -1;
			recent_pred = -1;
			//randomly select when there is a tie
			random_shuffle(act_set.begin(), act_set.end());
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				if (y[k] > max_y){
					recent_pred = k;
					max_y = y[k];
				}
			}
			//cerr << "recent_pred=" << recent_pred << ", c=" << c[recent_pred] << endl;
			return c[recent_pred];
		}
		
        inline Float rel_score(){
			Float score = 0.0;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				score += c[k]*y[k];
			}
			return score;
		}

		inline void display(){

			//cerr << grad[0] << " " << grad[1] << endl;
			cerr << endl;
			for (vector<int>::iterator it = act_set.begin(); it != act_set.end(); it++){
				int k = *it;
				cerr << k << ":" << y[k] << ":" << c[k] << " ";
			}
			cerr << endl;
			/*for (int k = 0; k < K; k++)
			  cerr << y[k] << " ";
			  cerr << endl;*/
		}

};

#endif
