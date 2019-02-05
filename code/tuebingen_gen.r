setwd("~/projects/randomized_causation_coefficient/code")
library(mixtools)

write('SampleID,A,B',file='syn_pairs.csv')

write_pair <- function(x,y) {
  write(paste(i, paste(x,collapse=" "), paste(y,collapse=" "), sep=","),
  append=TRUE, file="syn_pairs.csv")
}

cause <- function(n,k=5,p1=5,p2=5) {
  w <- abs(runif(k))
  w <- w/sum(w)
  m <- rnorm(k,0,p1)
  s <- abs(rnorm(k,1,p2))
  # s <- abs(rnorm(k,0,p2))  # Paper says mean 0
  scale(rnormmix(n,w,m,s))
}

noise <- function(n,v) {
  v*rnorm(n)
}

mechanism <- function(x,d=10) {
  g <- seq(min(x)-sd(x),max(x)+sd(x),length.out=d)  # Paper just says min to max
  function(z) predict(smooth.spline(g,rnorm(d)),z)$y
}

N  <- 1000
n <- 1000

set.seed(0)

# At various points in this file, they set c=5, sigma1=5, sigma2=5, sigma3=1, df=10
# Questionable because sigma1, sigma2, and df are at the edges of the opimization space

for(i in 1:N) {
  # x <- cause(1000)
  x <- cause(n)  # May as well generalize
  f <- mechanism(x)
  e <- noise(length(x),runif(1))  # Setting sigma3 = 1 implicitly
  write_pair(x,scale(f(x))+e)
}

write.table(cbind(1:(1*N),1,0),sep=",", quote=FALSE,
col.names=F,row.names=F,file="syn_target.csv")
