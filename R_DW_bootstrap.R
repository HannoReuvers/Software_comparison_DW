rm(list=ls())     # Clean memory
graphics.off()    # Close graphs
cat("\014")       # Clear Console

T= 500
Nsim= 1000
beta= c(1, 2)
presample= 50
B= 499

# List for theta
thetalist= seq(0,0.5,by=0.025)
L= length(thetalist)

# Generate signal
periodic<- sin(0.5*pi*seq(1, T))
mX<- cbind(rep(1,T),periodic)
signal= mX%*%beta

# Monte Carlo

reject_list<- matrix(0,nrow=Nsim,ncol=L)
ptm <- proc.time()
for (thetaiter in 1:L)
{
  print(thetaiter)
  for (MCiter in 1:Nsim)
  {
    eta_t= rnorm(T+presample,0,1)
    u_t= rep(0,T+presample)
    u_t[1]= eta_t[1]
    for (titer in 2:(T+presample))
    {
      u_t[titer]= thetalist[thetaiter]*u_t[titer-1]+eta_t[titer]
    }
    u_t= tail(u_t,T)
    
    # Estimate model
    vy= signal+u_t
    betahat= solve(crossprod(mX),crossprod(mX,vy))
    fit= mX%*%betahat
    uhat_t= vy-fit
    DW= sum(diff(uhat_t)^2)/sum(uhat_t^2)
    rhohat= (uhat_t[2:T]%*%uhat_t[1:T-1])/(uhat_t[1:T]%*%uhat_t[1:T])
    resid= uhat_t[2:T]-rhohat[1]%*%uhat_t[1:T-1]
    
    DW_star<- rep(0,B)
    for (biter in 1:B)
    {
      vy_star= fit+resid[ceiling(runif(T,0,T-1))]
      betahat_star= solve(crossprod(mX),crossprod(mX,vy_star))
      resid_star= vy_star-mX%*%betahat_star
      DW_star[biter]= sum(diff(resid_star)^2)/sum(resid_star^2)
    }
    
    # Evaluate hypothesis test
    crit_low= quantile(DW_star,0.025)
    crit_high= quantile(DW_star,0.975)
    
    if (DW<crit_low){
      reject_list[MCiter,thetaiter]= 1
    } else if (DW>crit_high) {
      reject_list[MCiter,thetaiter]=1
    }
  }
}
elapsed_time= proc.time() - ptm 
plot(thetalist,colMeans(reject_list))





