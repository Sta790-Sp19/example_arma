// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
std::vector<double> arma_stl(arma::mat M) {
  std::vector<double> v(M.begin(), M.end());
  
  return v;
}

/*** R

m = matrix(1:9, 3, 3)

arma_stl(m)
***/