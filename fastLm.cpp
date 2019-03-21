// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List fastLm(const arma::mat& X, const arma::colvec& y) {
  int n = X.n_rows, k = X.n_cols;
  
  arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
  arma::colvec res  = y - X*coef;           // residuals
  
  // std.errors of coefficients
  double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - k);
  
  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("stderr")       = std_err,
    Rcpp::Named("df.residual")  = n - k
  );
}

// [[Rcpp::export]]
Rcpp::List fastLm2(const arma::mat& X, const arma::colvec& y) {
  int n = X.n_rows, k = X.n_cols;
  
  arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
  arma::colvec res  = y - X*coef;           // residuals
  
  // std.errors of coefficients
  double s2 = arma::accu( arma::pow(res, 2.0) )/(n - k);
  
  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("stderr")       = std_err,
    Rcpp::Named("df.residual")  = n - k
  );
}

// [[Rcpp::export]]
Rcpp::List fastLm3(const arma::mat& X, const arma::colvec& y) {
  int n = X.n_rows, k = X.n_cols;
  
  arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
  arma::colvec res  = y - X*coef;           // residuals
  
  // std.errors of coefficients
  double s2 = arma::accu( res % res )/(n - k);
  
  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("stderr")       = std_err,
    Rcpp::Named("df.residual")  = n - k
  );
}

// [[Rcpp::export]]
Rcpp::List fastLm4(const arma::mat& X, const arma::colvec& y) {
  int n = X.n_rows, k = X.n_cols;
  
  arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
  arma::colvec res  = y - X*coef;           // residuals
  
  // std.errors of coefficients
  double s2 =  arma::as_scalar(res.t() * res) /(n - k);
  
  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("stderr")       = std_err,
    Rcpp::Named("df.residual")  = n - k
  );
}


/*** R
library(dplyr)
n=1000
d = tibble(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  x4 = rnorm(n),
  x5 = rnorm(n),
) %>%
  mutate(
    y = 3 + x1 - x2 + 2*x3 -2*x4 + 3*x5 - rnorm(n)
  )
X = model.matrix(y ~ ., d)
y = as.matrix(d$y)

bench::mark(
  fastLm(X, y),
  fastLm2(X, y),
  fastLm3(X, y),
  fastLm4(X, y)
)
*/
