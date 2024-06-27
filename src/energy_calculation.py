from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from abc import ABCMeta, abstractmethod

class EnergyCalculation(metaclass = ABCMeta):

    def __init__(self, target_speeds, acceleration_coeff) -> None:
        self.MAX_SPEED = np.max(target_speeds)
        self.MIN_SPEED = np.min(target_speeds)
        self.acceleration_coeff = acceleration_coeff

    @abstractmethod
    def compute_efficiency(self, vehicle: Vehicle, normalise: bool = True) -> float:
        return

class CO2EnergyCalculation(EnergyCalculation):
    '''Computes energy efficiency based on the formula in https://journals.sagepub.com/doi/full/10.1177/0361198119839970'''
    
    def __init__(self, target_speeds, acceleration_coeff) -> None:
        super().__init__(target_speeds, acceleration_coeff)
        self.MAX_ENERGY_CONSUMPTION  = self.__compute_max_energy_consumption()

    def compute_efficiency(self, vehicle: Vehicle, normalise: bool = True) -> float:
        ''' Calculates the difference between maximum and current CO2 emmissions, 
            which is taken as a means of measuring the energy efficiency.
            Meaning: poor energy efficiency --> close to 0, great efficiency --> close to value 1 (for normalise = True)
        '''

        curr_ac = vehicle.action['acceleration'] 
        curr_speed = vehicle.speed

        # maximum possible acceleration for the current speed
        # computed using the formula found in ControlledVehicle class of highway-env

        current_energy_consumption = CO2EnergyCalculation.__compute_co2_emission(curr_ac, curr_speed)

        efficiency = self.MAX_ENERGY_CONSUMPTION - current_energy_consumption
        
        # compute maximum possible energy consumption based on maximum velocity and acceleration 
        # and use this value for normalisation
        if normalise:
            efficiency /= self.MAX_ENERGY_CONSUMPTION

        return efficiency

    def __compute_max_energy_consumption(self, vehicle: Vehicle):
        '''Calculates the maximum possible energy consumption value of a given energy consumption function based
        on speed and acceleration values.'''
        speeds = vehicle.target_speeds
        max_energy = 0
        for speed in speeds:
            #compute max co2 emmission based on specific speed
            max_acc = vehicle.KP_A * (vehicle.MAX_SPEED - speed)
            energy = CO2EnergyCalculation.__compute_co2_emission(max_acc, speed)

            #update max value
            if energy > max_energy:
                max_energy = energy
        return max_energy

    def __compute_co2_emission(acceleration, velocity, type='light_passenger',fuel='gasoline'):
        '''Code taken from: https://github.com/amrzr/SA-MOEAMOPG/blob/55ceddc58062f2d7d26107d7813d2dd7328f2203/SAMOEA_PGMORL/highway_env/envs/two_way_env.py#L139C1-L209C22.
            Equation is described in: https://journals.sagepub.com/doi/full/10.1177/0361198119839970'''
        
        #clip acceleration to minimum value of 0 because breaking would 
        #result in the same instantateous energy efficiency as not accelerating
        if acceleration < 0:
            acceleration = 0

        if fuel == 'gasoline':
            T_idle = 2392    # CO2 emission from gasoline [gCO2/L]
            E_gas =  31.5e6  # Energy in gasoline [J\L]
        elif fuel == 'diesel':
            T_idle = 2660   # CO2 emission from diesel [gCO2/L]
            E_gas =  38e6   # Energy in diesel [J\L]

        if type == 'light_passenger':
            M = 1334    # light-duty passenger vehicle mass [kg]
        elif type == 'light_van':
            M = 1752    # light-duty van vehicle mass [kg]
        
        Crr = 0.015     # Rolling resistance
        Cd  = 0.3       # Aerodynamic drag coefficient
        A = 2.5         # Frontal area [m2]
        g = 9.81        # Gravitational acceleration
        r = 0           # Regeneration efficiency ratio
        pho = 1.225     # Air density
        fuel_eff = 0.7  # fuel efficiency [70%]

        
        condition = M  * acceleration * velocity + M  * g * Crr * velocity +0.5 * Cd * A  * pho * velocity **3
        
        Ei = T_idle  / E_gas  * condition

        if Ei <= 0:
            E = r
        else:
            Ei = Ei * (velocity + 0.5 * acceleration)
            E = Ei/fuel_eff

        return np.abs(E)

class NaiveEnergyCalculation(EnergyCalculation):
    '''Calculates the energy efficiency based on a more naive approach which limits the effect of the acceleration
    so that the normalised values are more evenly spread out'''
    
    def __init__(self, target_speeds, acceleration_coeff) -> None:
        super().__init__(target_speeds, acceleration_coeff)
        self.MIN_ENERGY_CONSUMPTION = NaiveEnergyCalculation.__compute_naive_energy_consumption(0, self.MIN_SPEED)
        self.MAX_ENERGY_CONSUMPTION = NaiveEnergyCalculation.__compute_naive_energy_consumption(np.inf, self.MAX_SPEED)

    def compute_efficiency(self, vehicle: Vehicle, normalise: bool = True) -> float:
        curr_ac = vehicle.action['acceleration']

        curr_energy_consumption = NaiveEnergyCalculation.__compute_naive_energy_consumption(curr_ac, vehicle.speed)
        curr_energy_consumption = max(curr_energy_consumption, self.MIN_ENERGY_CONSUMPTION)
        
        energy_efficiency = self.MAX_ENERGY_CONSUMPTION - curr_energy_consumption
        #normalisation
        if normalise:
            energy_efficiency = energy_efficiency / (self.MAX_ENERGY_CONSUMPTION - self.MIN_ENERGY_CONSUMPTION)

        return energy_efficiency

    def __compute_naive_energy_consumption(acceleration, velocity):
        acceleration = np.clip(acceleration,0,None)
        sigmoid_acc = 1/(1+np.exp(-acceleration))
        energy_consumption = (velocity ** 3) * sigmoid_acc

        return energy_consumption