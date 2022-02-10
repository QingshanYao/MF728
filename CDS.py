# author: Dante
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

class CDS:
    def __init__(self, cTenors, cSpreads, premiumFrequency, defaultFrequency, recoveryRate):

        self.cTenors = cTenors
        self.cSpreads = cSpreads
        self.premiumFrequency = premiumFrequency
        self.defaultFrequency = defaultFrequency
        self.recoveryRate = recoveryRate
        self.spreads = None

    def DF(self, yTenors, yRate, t):
        # yTenors: time points of yield curve
        # yRate: yield rates wrt yTenors

        result = -1
        left = 0
        right = len(yTenors) - 1

        if t < 0:
            result = - 1
        elif t == 0:
            result = 1.0
        elif t > 0 and t <= yTenors[left]:
            #result = 1/(1+yRate[0])**t
            result = np.exp(-t*yRate[0])
        elif t >= yTenors[right]:
            #result = 1/(1+yRate[-1])**t
            result = np.exp(-t*yRate[-1])
        else:
            for i in range(right):
                #find the interval where t lies
                if t >= yTenors[i] and t < yTenors[i+1]:
                    yield_sim = yRate[i] + (yRate[i+1] - yRate[i]) / (yTenors[i+1]-yTenors[i]) * (t-yTenors[i])
                    #result = 1/(1+yield_sim)**t
                    result = np.exp(-t*yield_sim)
        return result

    def Pt(self, cTenors, survProb, t):
        # cTenors: time points of cds curve
        # survProb: survival probability
        result = -1
        left = 0
        right = len(cTenors) - 1
        if t < 0:
            result = -1
        elif t == 0:
            result = 1
        elif t > 0 and t <= cTenors[left]:
            # h = hazard rate
            h = -np.log(survProb[0] / cTenors[left])
            result = np.exp(-h*t)
        elif t == cTenors[right]:
            result = survProb[-1]
        elif t > cTenors[right]:
            h = 0
            if len(cTenors) == 1:
                h =  - np.log(survProb[-1]) / cTenors[right]
            else:
                h = - np.log(survProb[-1]/survProb[-2]) / (cTenors[-1]-cTenors[-2])
                result = survProb[-1] * np.exp(-h*(t - cTenors[right]))
        else:  
            for i in range(right):
                #find the interval where t lies
                if t >= cTenors[i] and t < cTenors[i+1]:
                    h = -np.log(survProb[i+1]/survProb[i]) / (cTenors[i+1]-cTenors[i])
                    result = survProb[i] * np.exp(-h*(t-cTenors[i]))
        return result

    def PremiumLeg(self, cTenors, survProb, yTenors, yRate, maturity, num_count, spread, h):
        right = len(cTenors) - 1

        if right > 0 and maturity <= cTenors[right]:
            RPV01 = 0
            N = int(maturity*num_count)
            for n in range(1, N+1):
                tn = n / num_count
                tn_1 = (n-1) / num_count
                dt = 1.0 / num_count
                RPV01 += 0.5 * dt * self.DF(yTenors, yRate, tn) * (self.Pt(cTenors, survProb, tn) + self.Pt(cTenors, survProb, tn_1) )
            return spread * RPV01
        else:
            RPV01 = 0
            N = int(maturity * num_count)
            M = cTenors[right] * num_count if right >= 0 else 0
            for n in range(1, N+1):
                if n <= M:
                    tn = n/num_count
                    tn_1 = (n-1)/num_count
                    dt = 1.0 / num_count
                    RPV01 += 0.5 * dt * self.DF(yTenors, yRate, tn) * (self.Pt(cTenors, survProb, tn) + self.Pt(cTenors, survProb, tn_1) )
                else:
                    tn = n/num_count
                    tn_1 = (n-1)/num_count
                    tM = M / num_count
                    dt = 1.0 / num_count
                    Pt_n = self.Pt(cTenors, survProb, tM) * np.exp(-h*(tn - tM))
                    Pt_n_1 = 0
                    if tn_1 <= tM:
                        Pt_n_1 = self.Pt(cTenors, survProb, tn_1)
                    else:
                        Pt_n_1 = self.Pt(cTenors, survProb, tM) * np.exp(-h*(tn_1 - tM))

                    RPV01 += 0.5 * dt * self.DF(yTenors, yRate, tn) * (Pt_n + Pt_n_1)
            return spread * RPV01

    def DefaultLeg(self, cTenors, survProb, yTenors, yRate, maturity, num_count, RR, h):
        right = len(cTenors) - 1
        if right > 0 and maturity <= cTenors[right]:
            res = 0
            N = int(maturity * num_count)
            for n in range(1, N+1):
                tn = n / num_count
                tn_1 = (n-1) / num_count
                res += self.DF(yTenors, yRate, tn)*(self.Pt(cTenors, survProb, tn_1) - self.Pt(cTenors, survProb, tn))
            return (1 - RR) * res
        else:
            res = 0
            N = int(maturity*num_count)
            M = cTenors[right] * num_count if right >= 0 else 0

            for n in range(1, N+1):
                if n <= M:
                    tn = n / num_count
                    tn_1 = (n-1) / num_count
                    res += self.DF(yTenors, yRate, tn)*(self.Pt(cTenors, survProb, tn_1) - self.Pt(cTenors, survProb, tn))
                else:  # n > m
                    tM = M / num_count
                    tn = n / num_count
                    tn_1 = (n-1) / num_count

                    Pt_n = self.Pt(cTenors, survProb, tM) * np.exp(-h*(tn-tM))
                    if tn_1 <= tM:
                        Pt_n_1 = self.Pt(cTenors, survProb, tn_1)
                    else:
                        Pt_n_1 = self.Pt(cTenors, survProb,  tM) * np.exp(-h*(tn_1 - tM))
                    res += self.DF(yTenors, yRate, tn) * (Pt_n_1 - Pt_n)

            return (1-RR)*res

    def bootstrap(self, yTenors, yRate, cTenors, cSpreads):
        PF = self.premiumFrequency
        DF = self.defaultFrequency
        RR = self.recoveryRate

        def objfunFindHazardRate(h, survProb,  cTenors, maturity, spread):
            premLeg = self.PremiumLeg(cTenors, survProb, yTenors, yRate, maturity, PF, spread, h)
            defaultLeg = self.DefaultLeg(cTenors, survProb, yTenors, yRate, maturity, DF, RR, h)
            return premLeg - defaultLeg

        cnodes = len(cTenors)

        sp = [None]*cnodes
        hazardRate = [None]*cnodes

        SP = []
        CT = []
        for i in range(cnodes):

            maturity = cTenors[i]
            spread = cSpreads[i]

            h = newton(objfunFindHazardRate, 0, args=(SP, CT, maturity, spread),tol=1e-12, maxiter=100)
            hazardRate[i] = h
            if i == 0:
                sp[i] = np.exp(-hazardRate[i]*cTenors[i])
            else:
                sp[i] = sp[i-1] * np.exp(-hazardRate[i]*(cTenors[i]-cTenors[i-1]))
            CT.append(cTenors[i])
            SP.append(sp[i])
        return hazardRate, sp


if __name__ == "__main__":
    yieldcurveTenor = [1, 2, 3, 5]
    yieldcurveRate = [0.02, 0.02, 0.02, 0.02]
    cdsTenors = [1, 2, 3,  5]
    cdsSpreads = [0.010, 0.011, 0.012,  0.014]
    premiumFrequency = 4
    defaultFrequency = 4
    recoveryRate = 0.40
    cds_obj = CDS(cdsTenors,cdsSpreads,premiumFrequency,defaultFrequency,recoveryRate)
    # (a)
    # bootstrap the h and P
    h, P = cds_obj.bootstrap(yieldcurveTenor,yieldcurveRate,cdsTenors,cdsSpreads)

    print("hazard rate",h)
    print("survival probability",P)
    # get the spread of 4y CDS
    def objfunFindSpread(spread, survProb,  cTenors, maturity, h):
        premLeg = cds_obj.PremiumLeg(cdsTenors, P, yieldcurveTenor, yieldcurveRate, maturity, premiumFrequency, spread, h)
        defaultLeg = cds_obj.DefaultLeg(cTenors, survProb, yieldcurveTenor, yieldcurveRate, maturity, defaultFrequency, recoveryRate, h)
        return premLeg - defaultLeg
    spread_4 = newton(objfunFindSpread, 0.01, args=(P, cdsTenors, 4, h),tol=1e-12, maxiter=100)
    print("spread for 4y",spread_4)
    # get complete h and P table
    hc, pc = cds_obj.bootstrap(yieldcurveTenor,yieldcurveRate,[1,2,3,4,5],[0.01,0.011,0.012,spread_4,0.014])
    spread_list =[] 
    print("For all five CDS:")
    print("h rate:",hc)
    print("SurvProb", pc)

    # hazard rate curve
    h_list=[]
    for i in np.linspace(0.25,5,16):
        if i <=1:
            h_list.append(h[0])
        elif i <=2:
            h_list.append(h[1])
        elif i <=3:
            h_list.append(h[2])
        else:
            h_list.append(h[3])
    plt.plot(np.linspace(0.25,5,16), h_list)
    plt.xlabel("maturity / year")
    plt.ylabel("hazard rate")
    plt.show()        
    #bootstrap CDS curve
    print("Bootstrapped CDS spread curve(quarterly payment):")
    for i in np.linspace(0.25,5,16):
        spread = newton(objfunFindSpread, 0.01, args=(P, cdsTenors, i, h),tol=1e-12, maxiter=100)
        spread_list.append(spread)
    plt.plot(np.linspace(0.25,5,16), spread_list)
    plt.xlabel("maturity / year")
    plt.ylabel("spread / bps")
    plt.show()
    #(b) solved in (a)
    #(c) 
    # calculate RPV01(t)
    print("problem c:")
    RPV01_t = cds_obj.PremiumLeg([1,2,3,4,5],pc,yieldcurveTenor,yieldcurveRate,4,4,1,hc)
    print("RPV01:",RPV01_t)
    dleg_t = cds_obj.DefaultLeg([1,2,3,4,5], pc,yieldcurveTenor,yieldcurveRate,4,4,recoveryRate,hc)
    S_t = dleg_t/RPV01_t
    print("S_t:",S_t)
    print("MTM:",(S_t-0.008)*RPV01_t)

    #(d)
    print("problem d:")
    S1 = (0.0132527-spread_4)*RPV01_t
    # up_cds = [0.0101,0.0111,0.0121,0.0141]
    # c_up_h, c_up_P = cds_obj.bootstrap(yieldcurveTenor,yieldcurveRate,cdsTenors,up_cds)
    # c_up_spread_4 = newton(objfunFindSpread, 0.01, args=(c_up_P, cdsTenors, 4, c_up_h),tol=1e-12, maxiter=100)
    # New_RPV01 = cds_obj.PremiumLeg([1,2,3,5],c_up_P,yieldcurveTenor,yieldcurveRate,4,4,1,c_up_h)
    # S2 = (0.0132527-c_up_spread_4)*New_RPV01
    print("dv01 wrt CDS(fix the contract spread):")
    print("new fair spread:", spread_4+0.0001)
    print("change of MTM:",RPV01_t)
    # DV01 = []
    # print("plot dv01 for different contract spreads")
    # for contract in range(50,400):
    #     S1 = (contract/10000-spread_4)*RPV01_t
    #     up_cds = [0.0101,0.0111,0.0121,0.0141]
    #     c_up_h, c_up_P = cds_obj.bootstrap(yieldcurveTenor,yieldcurveRate,cdsTenors,up_cds)
    #     c_up_spread_4 = newton(objfunFindSpread, 0.01, args=(c_up_P, cdsTenors, 4, c_up_h),tol=1e-12, maxiter=100)
    #     New_RPV01 = cds_obj.PremiumLeg([1,2,3,5],c_up_P,yieldcurveTenor,yieldcurveRate,4,4,1,c_up_h)
    #     S2 = (contract/10000-c_up_spread_4)*New_RPV01
    #     DV01.append(-(S2-S1))
    # plt.plot(np.linspace(50,400,350), DV01)

    # plt.xlabel("contract spread / bps")
    # plt.ylabel("DV01")
    # plt.show()

    # (e)
    # S1 = (0.0132527-spread_4)*RPV01_t
    yieldcurveRate = [0.0201,0.0201,0.0201,0.0201]
    # y_up_h, y_up_P = cds_obj.bootstrap(yieldcurveTenor,up_yield,cdsTenors,cdsSpreads)
    y_up_spread_4 = newton(objfunFindSpread, 0.01, args=(P, cdsTenors, 4, h),tol=1e-12, maxiter=100)
    New_RPV01 = cds_obj.PremiumLeg([1,2,3,4,5],pc,yieldcurveTenor,yieldcurveRate,4,4,1,hc)
    S2 = (y_up_spread_4 - spread_4)*New_RPV01
    print("dv01 wrt interest rate(fix the contract spread):")
    print("new RPV01", New_RPV01)
    print("new fair spread:", y_up_spread_4)
    print("change of MTM:",S2*10000)
    # IRDV01 = []
    # print("plot dv01 for different contract spreads")
    # for contract in range(50,400):
    #     S1 = (contract/10000-spread_4)*RPV01_t
    #     up_yield = [0.0201,0.0201,0.0201,0.0201]
    #     y_up_h, y_up_P = cds_obj.bootstrap(yieldcurveTenor,up_yield,cdsTenors,cdsSpreads)
    #     y_up_spread_4 = newton(objfunFindSpread, 0.01, args=(y_up_P, cdsTenors, 4, y_up_h),tol=1e-12, maxiter=100)
    #     New_RPV01 = cds_obj.PremiumLeg([1,2,3,5],y_up_P,yieldcurveTenor,up_yield,4,4,1,y_up_h)
    #     S2 = (contract/10000-y_up_spread_4)*New_RPV01

    #     IRDV01.append(-(S2-S1))
    # plt.plot(np.linspace(50,400,350), IRDV01)

    # plt.xlabel("contract spread / bps")
    # plt.ylabel("IR DV01")
    # plt.show()

    # (f)
    # S1 = (0.0132527-spread_4)*RPV01_t
    yieldcurveRate = [0.020,0.020,0.020,0.020]
    recoveryRate = 0.4001
    cds_obj2 = CDS(cdsTenors,cdsSpreads,premiumFrequency,defaultFrequency,recoveryRate)
    # rh, rP = cds_obj2.bootstrap(yieldcurveTenor,yieldcurveRate,cdsTenors,cdsSpreads)
    r_spread_4 = newton(objfunFindSpread, 0, args=(P, cdsTenors, 4, h),tol=1e-15, maxiter=100)
    New_RPV01 = cds_obj2.PremiumLeg([1,2,3,4,5],pc,yieldcurveTenor,yieldcurveRate,4,4,1,hc)
    S2 = (r_spread_4-spread_4)*New_RPV01
    print("dv01 wrt recovery rate(fix the contract spread):")
    print("new RPV01", New_RPV01)
    print("new fair spread:",r_spread_4)
    print("change of MTM:",S2*10000)
    # print("plot dv01 for different contract spreads")
    # REC01 = []
    # for contract in range(50,400):
    #     S1 = (contract/10000-spread_4)*RPV01_t
    #     recoveryRate = 0.4001
    #     cds_obj2 = CDS(cdsTenors,cdsSpreads,premiumFrequency,defaultFrequency,recoveryRate)
    #     rh, rP = cds_obj2.bootstrap(yieldcurveTenor,yieldcurveRate,cdsTenors,cdsSpreads)
    #     r_spread_4 = newton(objfunFindSpread, 0, args=(rP, cdsTenors, 4, rh),tol=1e-15, maxiter=100)
    #     New_RPV01 = cds_obj2.PremiumLeg([1,2,3,5],y_up_P,yieldcurveTenor,yieldcurveRate,4,4,1,y_up_h)
    #     S2 = (contract/10000-r_spread_4)*New_RPV01

    #     REC01.append(-(S2-S1))
    # plt.plot(np.linspace(50,400,350), REC01)

    # plt.xlabel("contract spread / bps")
    # plt.ylabel("REC 01")
    # plt.show()
