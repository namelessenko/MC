import numpy as np
from mostovoysums import *

class MonteCarloMinimization:
    def __init__(self, initial_angles, temperature, iterations):
        self.phi_current, self.theta_current = initial_angles
        self.current_angles = np.concatenate((self.phi_current, self.theta_current))
        self.current_energy = self.energy(self.current_angles)  # Initial energy
        self.temperature = temperature
        self.iterations = iterations
        self.N = len(self.phi_current)

    def energy(self, angles):
      phiInit, thetaInit = angles.reshape(2,N,N)
      
      S = 1/2*np.array([np.sin(thetaInit)*np.cos(phiInit),
                    np.sin(thetaInit)*np.sin(phiInit),
                    np.cos(thetaInit)])
      
      Energy = -J1*np.sum(sumNN(X,Y,S))\
               +J2*np.sum(sumNNN(X,Y,S))\
               -h * np.sum(S[2])\
               -K * np.sum(S[2]**2)
      
      return Energy

    def run_simulation(self):
        for _ in range(self.iterations):
            # Choose a random spin site
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)

            # Propose a new set of spin angles
            phi_proposal = np.copy(self.phi_current)
            theta_proposal = np.copy(self.theta_current)
            phi_proposal[i, j] = np.random.uniform(low=0, high=2 * np.pi)
            theta_proposal[i, j] = np.random.uniform(low=0, high=2 * np.pi)
            proposed_angles = np.concatenate((phi_proposal, theta_proposal))

            # Calculate the energy difference
            delta_e = self.energy(proposed_angles) - self.current_energy

            # Accept or reject the proposed spin configuration
            if delta_e < 0 or np.exp(-delta_e / self.temperature) > np.random.uniform():
                self.phi_current = np.copy(phi_proposal)
                self.theta_current = np.copy(theta_proposal)
                self.current_angles = np.copy(proposed_angles)
                self.current_energy += delta_e
            self.temperature -= 1 / self.iterations

        return self.current_angles, self.current_energy


if __name__ == "__main__":
    #PARAMS
    J1 = 1
    J2 = 0.5
    N = 20
    K = 0.15
    h = 0.2
    
    #Grid
    X, Y = np.meshgrid(range(N), range(N))
    a = 2/np.sqrt(3)
    a1 = a*np.array([1/2,np.sqrt(3)/2])
    a2 = a*np.array([1/2,-np.sqrt(3)/2])
    
    q = np.pi/(N)
    Q1 = q*np.array([0,1])
    Q2 = q*np.array([np.sqrt(3)/2,-1/2])
    Q1Rij = np.dot(X,np.dot(a1,Q1))+np.dot(Y,np.dot(a2,Q1))
    Q2Rij = np.dot(X,np.dot(a1,Q2))+np.dot(Y,np.dot(a2,Q2))
    
    angles = np.array([Q1Rij, Q2Rij])
    
    minimizer = MonteCarloMinimization(angles,1,10000)
    minimizedAnglesMC, minEnMC = minimizer.run_simulation()
    minimizedAnglesMC = minimizedAnglesMC.flatten()
    minimizedPhiMC = minimizedAnglesMC[:N*N].reshape(N,N)
    minimizedThetaMC = minimizedAnglesMC[N*N:].reshape(N,N)
    minimizedSMC = np.array([np.sin(minimizedThetaMC)*np.cos(minimizedPhiMC),
                        np.sin(minimizedThetaMC)*np.sin(minimizedPhiMC),
                        np.cos(minimizedThetaMC)])
    print(minEnMC/(N*N))
    Smin_normed = 1-(minimizedSMC-np.min(minimizedSMC))/(np.max(minimizedSMC)-np.min(minimizedSMC))*2
    
    import matplotlib.pyplot as plt
    plt.figure()
    u = Smin_normed[0]
    v = Smin_normed[1]
    plt.contourf(X, Y, Smin_normed[2], cmap='rainbow')
    plt.quiver(X, Y, u, v)
    plt.show()