clc; clear variables;

T= 100;
Nsim= 1000;
beta= [1; 2];
presample= 50;
B= 499;

% List for theta
thetalist= 0:0.025:0.5;
L= length(thetalist);

% Generate signal
periodic= sin(0.5*pi*(1:T))';
mX= [ones([T 1]) periodic];
signal= mX*beta;

% Monte Carlo
reject_list= zeros(Nsim,L);
tic
for thetaiter= 1:L
    disp(thetaiter)
    for MCiter= 1:Nsim
        
        % Compute AR(1) innovations
        eta_t= normrnd(0,1,[T+presample 1]);
        u_t= zeros(T+presample,1);
        u_t(1)= eta_t(1);
        for titer= 2:(T+presample)
            u_t(titer)= thetalist(thetaiter)*u_t(titer-1)+eta_t(titer);
        end
        u_t= u_t(end-T+1:end);
        
        % Estimate model
        vy= signal+u_t;
        betahat= (mX'*mX)\(mX'*vy);
        fit= mX*betahat;
        uhat_t= vy-fit;
        DW= sum(diff(uhat_t).^2 )/sum(uhat_t.^2);
        rhohat= (uhat_t(2:T)'*uhat_t(1:T-1))/(uhat_t'*uhat_t);
        resid= uhat_t(2:T)-rhohat*uhat_t(1:T-1);
         
        % Bootstrap
        DW_star= zeros(B,1);
        for biter= 1:B
            vy_star= fit+resid(unidrnd(T-1,[T 1]));
            betahat_star= (mX'*mX)\(mX'*vy_star);
            resid_star= vy_star-mX*betahat_star;
            DW_star(biter)= sum(diff(resid_star).^2 )/sum(resid_star.^2);
        end
        
        % Evaluate hypothesis test
        crit_low= quantile(DW_star,0.025);
        crit_up= quantile(DW_star,0.975);
        
        if DW < crit_low
            reject_list(MCiter,thetaiter)= 1;
        elseif DW > crit_up
            reject_list(MCiter,thetaiter)= 1;
        end
    end  
end
toc

plot(thetalist,mean(reject_list,1),'o')

