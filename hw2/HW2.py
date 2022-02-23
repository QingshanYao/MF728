from cmath import exp
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

class  yield_curve_construction:
    def __init__(self, maturities, swap_rates, pay_freq = 2):

        self.maturities = maturities
        self.swap_rates = swap_rates
        self.pay_freq = pay_freq

    def FixedLeg(self, T, T_diff, D_list, forward_rate):
        fixed_coupon = self.swap_rates[T]
        additional_D = 0
        pre_D = D_list[-1] if len(D_list) > 0 else 1
        for i in np.linspace(1/self.pay_freq, T_diff, T_diff*self.pay_freq):
            additional_D += pre_D * np.exp(-forward_rate * i)
        fixed_leg = 1/self.pay_freq * fixed_coupon * (np.sum(D_list) + additional_D) 
        return fixed_leg

    def FloatLeg(self, T, T_diff, D_list, forward_rate):
        float_coupon = self.pay_freq * (np.exp(forward_rate/self.pay_freq) - 1)
        if T > 1:
            pre_float = 1/self.pay_freq * self.swap_rates[T-T_diff] * np.sum(D_list)
        else:
            pre_float = 0
        additional_D = 0
        pre_D = D_list[-1] if len(D_list) > 0 else 1
        for i in np.linspace(1/self.pay_freq, T_diff, T_diff*self.pay_freq):
            additional_D += pre_D * np.exp(-forward_rate * i)
        float_leg = pre_float + 1/self.pay_freq * float_coupon * additional_D
        return float_leg

    def bootstrap_forward(self):

        def objfunFindForwardRate(forward_rate, T, T_diff, D_list):
            fixleg = self.FixedLeg(T, T_diff, D_list, forward_rate)
            floatleg = self.FloatLeg(T, T_diff, D_list, forward_rate)
            return fixleg - floatleg

        D = []
        forward_list = []
        for i in range(len(self.maturities)):
            T = self.maturities[i]
            T_diff = T - self.maturities[i-1] if i > 0 else 1
            
            f_new = newton(objfunFindForwardRate, 0.03, args=(T, T_diff, D), tol=1e-12, maxiter=100)
            forward_list.append(np.float32(f_new))


            pre_D = D[-1] if len(D) > 0 else 1
            for j in np.linspace(1/self.pay_freq, T_diff, T_diff * self.pay_freq):
                D.append(pre_D * np.exp(-j * np.float32(f_new)))
        return forward_list, D

    def breakeven_swap_rate(self, f_list, D_list, T):
        fixed_leg = 1/self.pay_freq * np.sum(D_list[:T * self.pay_freq])
        float_leg = 0
        for t in np.linspace(0.5, T, 2 * T):
            year_flag = np.array(self.maturities)[np.array(self.maturities) >= t][0]
            index = np.where(np.array(self.maturities) == year_flag)
            float_coupon = self.pay_freq * (np.exp(1/self.pay_freq * f_list[index[0][0]]) - 1)
            float_leg += 1/self.pay_freq * float_coupon * D_list[int(2 * t - 1)]

        return float_leg / fixed_leg
    
    def zero_rate(self, D_list):
        temp = [-np.log(d) * 2 / (i + 1) for i, d in enumerate(D_list)]
        return temp


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    swap_maturity = [1, 2, 3, 4, 5, 7, 10, 30]
    swap_rate = [2.8438/100, 3.060/100, 3.126/100, 3.144/100, 3.150/100, 3.169/100, 3.210/100, 3.237/100]
    dic_swap = dict(zip(swap_maturity, swap_rate))
    pay_frequency = 2

    print("problem (a) (b) (c):")
    curve_instance = yield_curve_construction(swap_maturity, dic_swap, pay_frequency)
    f_points, discount_points= curve_instance.bootstrap_forward()
    print("forward rates:")
    print(f_points)
    f_piecewise = np.zeros(60)
    for i,j in enumerate(swap_maturity):
        if i==0:
            f_piecewise[i:j*2] = f_points[i]
        else:
            if i == j-1:
                f_piecewise[i*2:j*2] = f_points[i]

            else:
                f_piecewise[swap_maturity[i-1]*2:j*2] = f_points[i]

    plt.plot(np.linspace(0,30,60),f_piecewise, label="forward rate")
    plt.plot(swap_maturity, swap_rate, label="swap rate")
    plt.xlabel("Maturity / year")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

    print("\nproblem (d):")
    breakeven_15y = curve_instance.breakeven_swap_rate(f_points, discount_points, 15)
    print("breakeven swap rate of a 15Y swap:",breakeven_15y)

    print("\nproblem (e):")
    zr = curve_instance.zero_rate(np.array(discount_points))
    print("discount factors:")
    # print(discount_points)
    print(np.array(discount_points)[[1,3,5,7,9,13,19,59]])
    print("zero rates:")
    # print(zr)
    print(np.array(zr)[[1,3,5,7,9,13,19,59]])
    plt.plot(np.linspace(0,30,60),zr, label="zero rate")
    plt.plot(swap_maturity, swap_rate, label="swap rate")
    plt.xlabel("Maturity / year")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

    print("\nproblem (f):")
    new_f_points = np.array(f_points) + 0.01
    new_discount_points = [d * np.exp(-0.01 / pay_frequency * i) for i, d in enumerate(discount_points, start=1)]
    new_swap_rate = []
    for m in swap_maturity:
        temp_swap_rate = curve_instance.breakeven_swap_rate(new_f_points, new_discount_points, m)
        new_swap_rate.append(temp_swap_rate)
    print("origin swap rates:")
    print(swap_rate)
    print("new swap rates:")
    print(new_swap_rate)
    print("differences:")
    print(np.array(new_swap_rate)-np.array(swap_rate))
    plt.plot(swap_maturity, new_swap_rate, label="new swap rate")
    plt.plot(swap_maturity, swap_rate, label="swap rate")
    plt.xlabel("Maturity / year")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

    print("\nproblem (g) (h):")
    steep_swap_rate = [2.8438, 3.06, 3.126, 3.144 + 0.05, 3.15 + 0.1, 3.169 + 0.15, 3.21 + 0.25, 3.237 + 0.5]
    print("bear steep swap rates:")
    print(np.array(steep_swap_rate)/100)
    steep_dic_swap = dict(zip(swap_maturity, np.array(steep_swap_rate)/100))
    steep_curve_instance = yield_curve_construction(swap_maturity, steep_dic_swap, pay_frequency)
    steep_f_points, steep_discount_points= steep_curve_instance.bootstrap_forward()
    print("new forward rates:")
    print(steep_f_points)
    s_f_piecewise = np.zeros(60)
    for i,j in enumerate(swap_maturity):
        if i==0:
            s_f_piecewise[i:j*2] = steep_f_points[i]
        else:
            if i == j-1:
                s_f_piecewise[i*2:j*2] = steep_f_points[i]

            else:
                s_f_piecewise[swap_maturity[i-1]*2:j*2] = steep_f_points[i]

    plt.plot(np.linspace(0,30,60),s_f_piecewise, label="bear forward rate")
    plt.plot(np.linspace(0,30,60),f_piecewise, label="original forward rate")
    plt.xlabel("Maturity / year")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

    print("\nproblem (i) (j):")
    bull_swap_rate = [2.8438-0.5, 3.06-0.25, 3.126-0.15, 3.144 -0.1, 3.15 -0.05, 3.169, 3.21, 3.237 ]
    print("bull steep swap rates:")
    print(np.array(bull_swap_rate)/100)
    bull_dic_swap = dict(zip(swap_maturity, np.array(bull_swap_rate)/100))
    bull_curve_instance = yield_curve_construction(swap_maturity, bull_dic_swap, pay_frequency)
    bull_f_points, bull_discount_points= bull_curve_instance.bootstrap_forward()
    print("new forward rates:")
    print(bull_f_points)
    b_f_piecewise = np.zeros(60)
    for i,j in enumerate(swap_maturity):
        if i==0:
            b_f_piecewise[i:j*2] = bull_f_points[i]
        else:
            if i == j-1:
                b_f_piecewise[i*2:j*2] = bull_f_points[i]

            else:
                b_f_piecewise[swap_maturity[i-1]*2:j*2] = bull_f_points[i]

    plt.plot(np.linspace(0,30,60),b_f_piecewise, label="bull forward rate")
    plt.plot(np.linspace(0,30,60),f_piecewise, label="original forward rate")
    plt.xlabel("Maturity / year")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()