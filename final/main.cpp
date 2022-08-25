#include<mpi.h>
#include "tests/sc_header.h"
#include<bits/stdc++.h>
#include <omp.h>
using namespace std;
#define _IS_DEBUG 0
#if _IS_DEBUG
  #define N sc::n
  #define M sc::m
#else
  #define N sc::N_MAX
  #define M sc::M_MAX
#endif
int myid,num_procs;
uint32_t xs_y;
random_device rng;
uint32_t xorshift32() {
  xs_y ^= (xs_y << 13);xs_y ^= (xs_y >> 17);
  return xs_y ^= (xs_y << 5);
}
bitset<sc::M_MAX*(sc::M_MAX-1)/2> Board[sc::N_MAX],ALL_Board;
int Board_check_cnt[sc::N_MAX],ALL_Board_check_cnt;
void make_Board(){
  static bool is_accepted[sc::N_MAX][sc::M_MAX];
  #pragma omp parallel for
  for(int i=0;i<N;i++){
    //状態iから文字列w[j]に沿って移動する
    for(int j=0;j<M;j++){
      int now_itr = i;
      //遷移
      for(int k=0;k<sc::w_len[j];k++) now_itr = sc::T[sc::w[j][k]=='b'][now_itr]-1;
      is_accepted[i][j] = sc::F[now_itr];
    }
    //checkを埋める
    int bitset_itr = 0;
    for(int j=1;j<M;j++){
      for(int k=0;k<j;k++) Board[i].set(bitset_itr++,(is_accepted[i][j]!=is_accepted[i][k]));
    }
    Board_check_cnt[i] = Board[i].count();
  }
#if _IS_DEBUG
  fprintf(stderr,"DEBUG\n");
  for(int i=0;i<N;i++){
    fprintf(stderr,"状態%d\n",i);
    for(int j=0;j<M;j++){
      for(int k=0;k<j;k++) fprintf(stderr,"%d",Board[i].test(j*(j-1)/2+k));
      fprintf(stderr,"\n");
    }
  }
#endif
}
void make_ALL_Board(){
  //全ての状態の論理和
  for(int i=0;i<N;i++) ALL_Board |= Board[i];
  ALL_Board_check_cnt = ALL_Board.count();
}
struct state{
  int check_cnt = -1;
  bitset<sc::M_MAX*(sc::M_MAX-1)/2> check_set;
  bitset<sc::N_MAX> situation_set;
  constexpr bool operator>(const state &rhs) const noexcept {return this->check_cnt > rhs.check_cnt;}
};
int output_k,output_qs[sc::N_MAX];
int order[sc::N_MAX];
constexpr int MAX_PROCS = 16;
constexpr int PROCS = 4;
constexpr int SEARCH_LOOP = 10;
bitset<sc::N_MAX> ans_save[SEARCH_LOOP];
int score[SEARCH_LOOP*MAX_PROCS];
bool situation_save[sc::N_MAX];
bool proc_situation_save[PROCS*sc::N_MAX];
void make_order(){
  static bool is_first = true;
  if(is_first){
    iota(order,order+N,0);
    is_first = false;
    if(myid){
      for(int i=0;i<N-1;i++) swap(order[i],order[i+xorshift32()%(N-i)]);
    }
    return;
  }
  else{
    for(int i=0;i<N-1;i++) swap(order[i],order[i+xorshift32()%(N-i)]);
  }
}
void make_order2(bitset<sc::N_MAX> &ans){
  int itr = 0;
  for(int i=0;i<N;i++){
    if(ans.test(order[i])) swap(order[itr++],order[i]);
  }
  int now_k = ans.count();
  for(int i=0;i<now_k-1;i++) swap(order[i],order[i+xorshift32()%(now_k-i)]);
}
void beam_search(int SITUATION_SIZE,int BEAM_SIZE,bitset<sc::N_MAX> &ans){
  constexpr int MAX_BEAM_SIZE = 10;//ビーム幅
  constexpr int MAX_SITUATION_CNT_LIMIT = 100;//あらかじめ必要な状態の個数の上限を割り切る
  static state dp[MAX_SITUATION_CNT_LIMIT+1][MAX_BEAM_SIZE*2];
  static bool is_first = true;
  int ans_situation_cnt = ans.count();//ansの状態集合の要素数
  if(is_first){
    is_first = false;
  }
  else{
    for(int i=0;i<min(MAX_SITUATION_CNT_LIMIT+1,ans_situation_cnt);i++) fill(dp[i],dp[i]+BEAM_SIZE*2,state());
  }
  dp[0][0].check_cnt = 0;
  for(int i=0;i<SITUATION_SIZE;i++){
    int id = order[i];
    int LIMIT = min({i+1,MAX_SITUATION_CNT_LIMIT,ans_situation_cnt-1})+1;
    #pragma omp parallel for
    for(int j=1;j<LIMIT;j++){
      for(int k=BEAM_SIZE;k<BEAM_SIZE*2;k++){
        if(dp[j-1][k-BEAM_SIZE].check_cnt == -1) continue;
        dp[j][k] = dp[j-1][k-BEAM_SIZE];
        dp[j][k].situation_set.set(id);
        dp[j][k].check_set |= Board[id];
        dp[j][k].check_cnt = dp[j][k].check_set.count();
        if(dp[j][k].check_cnt == ALL_Board_check_cnt && j < ans_situation_cnt){
          #pragma omp critical
          {
            if(j < ans_situation_cnt){
              dp[j][k].check_cnt = -1;
              ans = dp[j][k].situation_set;
              ans_situation_cnt = j;
            }
          }
        }
      }
    }
    LIMIT = min({i+1,MAX_SITUATION_CNT_LIMIT,ans_situation_cnt-1})+1;
    #pragma omp parallel for
    for(int j=1;j<LIMIT;j++){
      nth_element(dp[j],dp[j]+BEAM_SIZE-1,dp[j]+BEAM_SIZE*2,greater<>());
    }
  }
}
void write_situation(bitset<sc::N_MAX> &ans){
  for(int i=0;i<N;i++){
    situation_save[i] = ans.test(i);
  }
}
void write_ans(int best_itr){
  best_itr %= num_procs;
  output_k = 0;
  for(int i=0;i<N;i++){
    if(proc_situation_save[best_itr*N+i]) output_qs[output_k++] = i+1;
  }
}
int choose_best_itr(int loop){
  int ans_score = INT_MAX,ans_itr = -1;
  for(int i=0;i<(loop+1)*num_procs;i++){
    int now_score = score[i];
    if(now_score < ans_score){
      ans_score = now_score;
      ans_itr = i;
    }
  }
  return ans_itr;
}
void calc_score(int loop){
  for(int i=0;i<num_procs;i++){
    int now_score = 0;
    for(int j=0;j<N;j++){
      if(proc_situation_save[i*N+j]) now_score += N*N+j;
    }
    score[loop*num_procs+i] = now_score;
  }
}
int main(int argc, char** argv) {
  sc::initialize(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  make_Board();
  fprintf(stderr,"id=%d:make_Board() done:%lf\n",myid,sc::get_elapsed_time());
  make_ALL_Board();
  fprintf(stderr,"id=%d:make_ALL_Board() done:%lf\n",myid,sc::get_elapsed_time());fflush(stderr);
  xs_y = rng();
  int bef_best_itr = -1;
  for(int loop=0;loop<SEARCH_LOOP;loop++){
    ans_save[loop].set();
    make_order();
    for(int i=0;i<8;i++){
      beam_search(ans_save[loop].count(),loop+1,ans_save[loop]);
      if(i<7) make_order2(ans_save[loop]);
    }
    fprintf(stderr,"id=%d:beam_search() done:%lf:k=%zu\n",myid,sc::get_elapsed_time(),ans_save[loop].count());fflush(stderr);
    write_situation(ans_save[loop]);
    MPI_Gather(situation_save,N,MPI_CXX_BOOL,proc_situation_save,N,MPI_CXX_BOOL,0,MPI_COMM_WORLD);
    if(myid==0){
      calc_score(loop);
      int best_itr = choose_best_itr(loop);
      if(bef_best_itr != best_itr){
        write_ans(best_itr);
        sc::output(output_k,output_qs);
        bef_best_itr = best_itr;
      }
    }
  }
  sc::finalize();
}
