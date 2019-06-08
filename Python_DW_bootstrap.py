import numpy as np
import matplotlib.pyplot as plt
import time

T = 500
Nsim = 1000
beta = [[1], [2]]
presample = 50
B = 499

# List for theta
thetalist = np.linspace(0, 0.5, num=21)
L = 21

# Generate signal
periodic = np.sin(0.5*np.pi*np.linspace(1, T, T))
mX = np.c_[np.ones(T), periodic]
signal = np.dot(mX, beta)

# Monte Carlo
reject_list = np.zeros((Nsim, L))
start_timer = time.time()
for thetaiter in range(L):
    print(thetaiter)
    for MCiter in range(Nsim):

        # Compute AR(1) innovations
        eta_t = np.random.normal(0, 1, (T+presample, 1))
        u_t = np.zeros((T+presample, 1))
        u_t[0] = eta_t[0]
        for titer in range(1, T+presample):
            u_t[titer] = thetalist[thetaiter]*u_t[titer-1]+eta_t[titer]
        u_t = u_t[presample:(T+presample)]

        # Estimate model
        vy = signal+u_t
        betahat = np.linalg.solve(np.dot(mX.T, mX), np.dot(mX.T, vy))
        fit = np.dot(mX, betahat)
        uhat_t = vy-fit
        DW = np.sum(np.power(np.diff(uhat_t, axis=0), 2))/np.sum(np.power(uhat_t, 2))
        rhohat = sum(uhat_t[:1]*uhat_t[:(T-1)])/sum(uhat_t*uhat_t)
        resid = uhat_t[:1]-rhohat*uhat_t[:(T-1)]

        # Bootstrap
        DW_star = np.zeros((B, 1))
        for biter in range(B):
            vy_star = fit+resid[np.random.randint(T-1, size=T)]
            betahat_star = np.linalg.solve(np.dot(mX.T, mX), np.dot(mX.T, vy_star))
            resid_star = vy_star-np.dot(mX, betahat_star)
            DW_star[biter] = np.sum(np.power(np.diff(resid_star, axis=0), 2))/np.sum(np.power(resid_star, 2))

        # Evaluate hypothesis test
        crit_low = np.quantile(DW_star, 0.025)
        crit_up = np.quantile(DW_star, 0.975)

        if DW < crit_low:
            reject_list[MCiter, thetaiter] = 1
        elif DW > crit_up:
            reject_list[MCiter, thetaiter] = 1
end_timer = time.time()

print(end_timer-start_timer)

plt.plot(thetalist, np.mean(reject_list, axis=0), 'o')
plt.show()
