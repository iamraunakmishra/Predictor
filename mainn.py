import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#--1 The Black Scholes Calculator --#
def black_scholes(S,K,T,r,sigma, option_type='call'):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1- sigma*np.sqrt(T)
 
    if option_type== 'call':
        price=S*norm.cdf(d1)- K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

    vega= S*norm.pdf(d1)*np.sqrt(T)    
    return price, vega 

#--2 Newton Raphson Method --#
def find_implied_vol(market_price, S, K , T , r ):
    sigma=0.3
    for i in range(100):
        price, vega=black_scholes(S,K,T,r,sigma)
        diff=market_price-price
        if abs(diff)<0.0001:
            return sigma
        sigma=sigma + diff/vega
    return sigma

#--3 Monte Carlo Simulator --#
    
def monte_carlo_simulator(S, T, r, sigma, iterations=10000):
    dt=T
    z=np.random.standard_normal(iterations)
    ST=S*np.exp((r-0.5*sigma**2)* dt + sigma*np.sqrt(dt)*z)
    return ST

#-- Running the Project --#
spot=24500
strike=24600
expiry=30/365
rate=0.07
market_p=350

# A Calculate IV #
iv=find_implied_vol(market_p, spot, strike ,expiry, rate)
print(f"1. Market's Implied Volatility :{iv*100:.2f}%")

# B Run Simulation #
final_prices=monte_carlo_simulator(spot, expiry, rate , iv)

# C Plotting the Results #

plt.hist(final_prices, bins=50, color='skyblue', edgecolor='black')
plt.axvline(strike, color='red', linestyle='dashed', label='Strike Price')
plt.title(f"Monte Carlo :10000 Possible Nifty Prices at Expiry")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()