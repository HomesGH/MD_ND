import pint

class UnitConverter:
    '''
    Class to convert simulation results which are typically in reduced units to SI units.
    '''
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

        # Reference units
        self.lengthRef = 1.0
        self.energyRef = 1.0
        self.massRef = 1.0

        # with constants
        self.lengthRef_u = self.lengthRef*self.ureg.angstrom
        self.energyRef_u = self.energyRef*self.ureg.k_B*self.ureg.degree_Kelvin
        self.massRef_u = self.massRef*self.ureg.atomic_mass_constant

        # Base units
        self.base_units = {
            '[length]': self.lengthRef_u,
            '[temperature]': self.energyRef_u/self.ureg.k_B,
            '[mass]': self.massRef_u,
            '[time]': self.lengthRef_u*((self.massRef_u/self.energyRef_u)**0.5),
            '[substance]': 1/self.ureg.N_A
        }

    def SI2reduced(self, value: float, unit_str: str) -> float:
        '''Convert a value in SI unit to reduced (dimensionless) units.'''
        quantity = self.Q_(value, unit_str)
        dimensions = quantity.dimensionality

        reduced_quantity = quantity

        for dim, exponent in dimensions.items():
            natural_unit = self.base_units[dim]
            reduced_quantity /= natural_unit ** exponent

        return reduced_quantity.to_base_units().magnitude


    def reduced2SI(self, value: float, unit_str: str) -> float:
        '''Convert a value in reduced (dimensionless) units to SI units.'''
        quantity = self.Q_(1, unit_str)
        dimensions = quantity.dimensionality

        si_quantity = self.Q_(value, '')

        for dim, exponent in dimensions.items():
            natural_unit = self.base_units[dim]
            si_quantity *= natural_unit ** exponent

        return si_quantity.to(unit_str).magnitude

# Expose methods globally for direct import
unitconverter = UnitConverter()
SI2reduced = unitconverter.SI2reduced
reduced2SI = unitconverter.reduced2SI

if __name__ == '__main__':
    print('Testing conversion')

    test_quants = {'mol/L': 25,
                'kPa': 544,
                'K': 312,
                'mol/(m^2*s)': 3216,
                'J/mol': 112,
                'MPa': 441,
                'm/s': 78,
                'kg': 3.18e-27,
                'm': 5.8e-9,
                '(g*s)/(J*m)': 0.681,
                'Pa*s': 0.973,
                'N*s/m^2': 0.973,  # Same as Pa*s
                'W/(m*K)': 91.3,
                'J/(s*m*K)': 91.3, # Same as W/(m*K)
                }

    for unit,value in test_quants.items():
        red_value = SI2reduced(value,unit)
        value_SI = reduced2SI(red_value,unit)
        print(f'{value} {unit} in reduced units: {red_value:.6} ;'
              f' and back: {value_SI:.4}')
        if abs((value-value_SI)/value)>1e-9:
            print('WARNING! Conversion not correct')
