from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cplex
import docplex.mp.model as cpx
from ortools.linear_solver import pywraplp
import time


def my_timer(f):
    def tmp(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        delta_time = time.time() - start_time
        print('Время выполнения функции {}'.format(delta_time))
        return result

    return tmp


from pulp import *


class Battery(object):
    """ Used to store information about the battery.
       :param current_charge: is the initial state of charge of the battery
       :param capacity: is the battery capacity in Wh
       :param charging_power_limit: the limit of the power that can charge the battery in W
       :param discharging_power_limit: the limit of the power that can discharge the battery in W
       :param battery_charging_efficiency: The efficiecny of the battery when charging
       :param battery_discharing_efficiecny: The discharging efficiency
    """

    def __init__(self,
                 current_charge=0.0,
                 capacity=0.0,
                 charging_power_limit=1.0,
                 discharging_power_limit=-1.0,
                 charging_efficiency=0.95,
                 discharging_efficiency=0.95):
        self.current_charge = current_charge
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency


class BatteryController(object):
    step = 960

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):

        self.step -= 1
        if (self.step == 1): return 0
        if (self.step > 1): number_step = min(96, self.step)
        #
        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()
        #
        energy = [None] * number_step

        for i in range(number_step):
            if pv_forecast[i] >= 50:
                energy[i] = load_forecast[i] - pv_forecast[i]
            else:
                energy[i] = load_forecast[i]
        # battery
        capacity = battery.capacity
        charging_efficiency = battery.charging_efficiency
        discharging_efficiency = 1. / battery.discharging_efficiency
        current = capacity * battery.current_charge
        limit = battery.charging_power_limit
        dis_limit = battery.discharging_power_limit
        limit /= 4.
        dis_limit /= 4.

        # Ortools
        solver = pywraplp.Solver.CreateSolver('CBC')    # CBC CLP GLOP SCIP

        # Variables: all are continuous
        charge = [solver.NumVar(0.0, limit, "c" + str(i)) for i in range(number_step)]
        dis_charge = [solver.NumVar(dis_limit, 0.0, "d" + str(i)) for i in range(number_step)]
        battery_power = [solver.NumVar(0.0, capacity, "b" + str(i)) for i in range(number_step + 1)]
        grid_buy = [solver.NumVar(0.0, solver.infinity(), "gb" + str(i)) for i in range(number_step)]
        grid_sell = [solver.NumVar(0.0, solver.infinity(), "gs" + str(i)) for i in range(number_step)]
        mu = [solver.IntVar(0, 1, "mu" + str(i)) for i in range(number_step)]
        # Objective function
        objective = solver.Objective()
        for i in range(number_step):
            objective.SetCoefficient(grid_buy[i], price_buy[i])
            objective.SetCoefficient(grid_sell[i], -price_sell[i])
        objective.SetMinimization()

        # 3 Constraints
        c_grid = [None] * number_step
        c_power = [None] * (number_step + 1)

        # first constraint
        c_power[0] = solver.Constraint(current, current)
        c_power[0].SetCoefficient(battery_power[0], 1)

        for i in range(0, number_step):
            # second constraint
            c_grid[i] = solver.Constraint(energy[i], solver.infinity())
            c_grid[i].SetCoefficient(grid_buy[i], 1)
            c_grid[i].SetCoefficient(grid_sell[i], -1)
            c_grid[i].SetCoefficient(charge[i], -1)
            c_grid[i].SetCoefficient(dis_charge[i], -1)
            # third constraint
            c_power[i + 1] = solver.Constraint(0, 0)
            c_power[i + 1].SetCoefficient(charge[i], charging_efficiency)
            c_power[i + 1].SetCoefficient(dis_charge[i], discharging_efficiency)
            c_power[i + 1].SetCoefficient(battery_power[i], 1)
            c_power[i + 1].SetCoefficient(battery_power[i + 1], -1)
            solver.Add(charge[i] <= mu[i] * limit)
            solver.Add((1 - mu[i]) * dis_limit <= dis_charge[i])
        # solve the model
        solver.Solve()

        return battery_power[1].solution_value() / capacity


class BatteryController_pulp(object):
    step = 960

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):

        self.step -= 1
        if (self.step == 1): return 0
        if (self.step > 1): number_step = min(96, self.step)
        #
        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()
        #
        energy = [None] * number_step

        for i in range(number_step):
            if pv_forecast[i] >= 50:
                energy[i] = load_forecast[i] - pv_forecast[i]
            else:
                energy[i] = load_forecast[i]
        # battery
        capacity = battery.capacity
        charging_efficiency = battery.charging_efficiency
        discharging_efficiency = 1. / battery.discharging_efficiency
        current = capacity * battery.current_charge
        limit = battery.charging_power_limit
        dis_limit = battery.discharging_power_limit
        limit /= 4.
        dis_limit /= 4.
        # if x is Continuous

        model = LpProblem("Smart energy", LpMinimize)
        # model = cpx.Model(name="MIP Model")

        # Описываем переменные
        charge = {i: LpVariable(name="c" + str(i), lowBound=0.0, upBound=limit) for i in range(number_step)}
        dis_charge = {i: LpVariable(name="d" + str(i), lowBound=dis_limit, upBound=0.0) for i in range(number_step)}
        battery_power = {i: LpVariable(name="b" + str(i), lowBound=0.0, upBound=capacity) for i in range(number_step + 1)}
        grid_buy = {i: LpVariable(name="gb" + str(i), lowBound=0.0) for i in range(number_step)}
        grid_sell = {i: LpVariable(name="gs" + str(i), lowBound=0.0) for i in range(number_step)}
        mu = {i: LpVariable(name="mu" + str(i), lowBound=0.0, upBound=1, cat=LpInteger) for i in range(number_step)}

        model += lpSum([grid_buy[i] * price_buy[i] - grid_sell[i] * price_sell[i] for i in range(number_step)])

        model += battery_power[0] == current

        for i in range(0, number_step):
            model += grid_buy[i] - grid_sell[i] - charge[i] - dis_charge[i] >= energy[i]
            model += charge[i] * charging_efficiency + dis_charge[i] * discharging_efficiency + battery_power[i] == battery_power[i + 1]
            model += charge[i] <= mu[i] * limit
            model += dis_charge[i] >= (1 - mu[i]) * dis_limit

        model.solve(CPLEX_PY(msg=0))
        return battery_power[1].value() / capacity

class BatteryController_docplex(object):
    step = 960

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):

        self.step -= 1
        if self.step == 1: return 0
        if (self.step > 1): number_step = min(96, self.step)
        # if self.step > 1: number_step = 96

        price_buy = price_buy.tolist()
        price_sell = price_sell.tolist()
        load_forecast = load_forecast.tolist()
        pv_forecast = pv_forecast.tolist()
        #
        energy = [None] * number_step

        for i in range(number_step):
            if (pv_forecast[i] >= 50):
                energy[i] = load_forecast[i] - pv_forecast[i]
            else:
                energy[i] = load_forecast[i]
        # battery
        capacity = battery.capacity
        charging_efficiency = battery.charging_efficiency
        discharging_efficiency = 1. / battery.discharging_efficiency
        current = capacity * battery.current_charge
        limit = battery.charging_power_limit
        dis_limit = battery.discharging_power_limit
        limit /= 4.
        dis_limit /= 4.

        # CPLEX
        opt_model = cpx.Model(name="MIP Model")
        # Variables: all are continuous

        charge = {i: opt_model.continuous_var(lb=0.0,
                                              ub=limit,
                                              name="c_{0}".format(i))
                  for i in range(number_step)}
        dis_charge = {i: opt_model.continuous_var(lb=dis_limit,
                                                  ub=0.0,
                                                  name="d_{0}".format(i))
                      for i in range(number_step)}
        battery_power = {i: opt_model.continuous_var(lb=0.0,
                                                     ub=capacity,
                                                     name="b_{0}".format(i))
                         for i in range(number_step + 1)}
        grid_sell = {i: opt_model.continuous_var(lb=0.0,
                                         ub=None,
                                         name="gs_{0}".format(i))
                for i in range(number_step)}
        grid_buy = {i: opt_model.continuous_var(lb=0.0,
                                                 ub=None,
                                                 name="gb_{0}".format(i))
                     for i in range(number_step)}

        mu = {i: opt_model.binary_var(name="mu_{0}".format(i)) for i in range(number_step)}
        # == constraints
        # first constraint

        constraints = {0: opt_model.add_constraint(
            ct=opt_model.sum(battery_power[0]) == current,
            ctname="constraint_{0}".format(0))
        }
        # third constraint
        constraints.update({i + 1: opt_model.add_constraint(
            ct=opt_model.sum(charge[i] * charging_efficiency + dis_charge[i] * discharging_efficiency + battery_power[i] - battery_power[i + 1]) == 0,
            ctname="constraint_{0}".format(i + 1)) for i in range(0, number_step)
        })

        # second constraint
        constraints.update({number_step + 1 + i: opt_model.add_constraint(
            ct=opt_model.sum(grid_buy[i] - grid_sell[i] - charge[i] - dis_charge[i]) >= energy[i],
            ctname="constraint_{0}".format(number_step + 1 + i))
            for i in range(0, number_step)})
        # 0<=charge[i]<=mu[i]*limit
        constraints.update({2 * number_step + 1 + i: opt_model.add_constraint(
            ct=opt_model.sum(charge[i]) <= mu[i] * limit,
            ctname="constraint_{0}".format(2 * number_step + 1 + i))
            for i in range(0, number_step)})
        # 1-mu[i])*dis_limit <= dis_charge[i] <= 0
        constraints.update({3 * number_step + 1 + i: opt_model.add_constraint(
            ct=opt_model.sum(dis_charge[i]) >= (1 - mu[i]) * dis_limit,
            ctname="constraint_{0}".format(3 * number_step + 1 + i))
            for i in range(0, number_step)})

        objective = opt_model.sum(grid_buy[i] * price_buy[i] - grid_sell[i] * price_sell[i] for i in range(number_step))

        opt_model.minimize(objective)
        res = opt_model.solve()

        return res[battery_power[1]] / capacity



class Simulation(object):
    """ Handles running a simulation.
    """

    def __init__(self,
                 data,
                 battery,
                 site_id):
        """ Creates initial simulation state based on data passed in.
            :param data: contains all the time series needed over the considered period
            :param battery: is a battery instantiated with 0 charge and the relevant properties
            :param site_id: the id for the site (building)
        """

        self.data = data

        # building initialization
        self.actual_previous_load = self.data.actual_consumption.values[0]
        self.actual_previous_pv = self.data.actual_pv.values[0]

        # align actual as the following, not the previous 15 minutes to
        # simplify simulation
        self.data.loc[:, 'actual_consumption'] = self.data.actual_consumption.shift(-1)
        self.data.loc[:, 'actual_pv'] = self.data.actual_pv.shift(-1)

        self.site_id = site_id
        self.load_columns = data.columns.str.startswith('load_')
        self.pv_columns = data.columns.str.startswith('pv_')
        self.price_sell_columns = data.columns.str.startswith('price_sell_')
        self.price_buy_columns = data.columns.str.startswith('price_buy_')

        # initialize money at 0.0
        self.money_spent = 0.0
        self.money_spent_without_battery = 0.0

        # battery initialization
        self.battery = battery

    def run(self):
        """ Executes the simulation by iterating through each of the data points
            It returns both the electricity cost spent using the battery and the
            cost that would have been incurred with no battery.
        """
        battery_controller = BatteryController_pulp()                # choose here class of solvers

        for current_time, timestep in tqdm(self.data.iterrows(), total=self.data.shape[0], desc=' > > > > timesteps\t'):
            # can't calculate results without actual, so skip (should only be last row)
            if pd.notnull(timestep.actual_consumption):
                self.simulate_timestep(battery_controller, current_time, timestep)

        return self.money_spent, self.money_spent_without_battery

    def simulate_timestep(self, battery_controller, current_time, timestep):
        """ Executes a single timestep using `battery_controller` to get
            a proposed state of charge and then calculating the cost of
            making those changes.
            :param battery_controller: The battery controller
            :param current_time: the timestamp of the current time step
            :param timestep: the data available at this timestep
        """
        # get proposed state of charge from the battery controller
        proposed_state_of_charge = battery_controller.propose_state_of_charge(
            self.site_id,
            current_time,
            deepcopy(self.battery),
            self.actual_previous_load,
            self.actual_previous_pv,
            timestep[self.price_buy_columns],
            timestep[self.price_sell_columns],
            timestep[self.load_columns],
            timestep[self.pv_columns]
        )

        # get energy required to achieve the proposed state of charge
        grid_energy, battery_energy_change = self.simulate_battery_charge(self.battery.current_charge,
                                                                          proposed_state_of_charge,
                                                                          timestep.actual_consumption,
                                                                          timestep.actual_pv)

        grid_energy_without_battery = timestep.actual_consumption - timestep.actual_pv

        # buy or sell energy depending on needs
        price = timestep.price_buy_00 if grid_energy >= 0 else timestep.price_sell_00
        price_without_battery = timestep.price_buy_00 if grid_energy_without_battery >= 0 else timestep.price_sell_00

        # calculate spending based on price per kWh and energy per Wh
        self.money_spent += grid_energy * (price / 1000.)
        self.money_spent_without_battery += grid_energy_without_battery * (price_without_battery / 1000.)

        # update current state of charge
        self.battery.current_charge += battery_energy_change / self.battery.capacity
        self.actual_previous_load = timestep.actual_consumption
        self.actual_previous_pv = timestep.actual_pv

    def simulate_battery_charge(self, initial_state_of_charge, proposed_state_of_charge, actual_consumption, actual_pv):
        """ Charges or discharges the battery based on what is desired and
            available energy from grid and pv.
            :param initial_state_of_charge: the current state of the battery
            :param proposed_state_of_charge: the proposed state for the battery
            :param actual_consumption: the actual energy consumed by the building
            :param actual_pv: the actual pv energy produced and available to the building
        """
        # charge is bounded by what is feasible
        proposed_state_of_charge = np.clip(proposed_state_of_charge, 0.0, 1.0)

        # calculate proposed energy change in the battery
        target_energy_change = (proposed_state_of_charge - initial_state_of_charge) * self.battery.capacity

        # efficiency can be different whether we intend to charge or discharge
        if target_energy_change >= 0:
            efficiency = self.battery.charging_efficiency
            target_charging_power = target_energy_change / ((15. / 60.) * efficiency)
        else:
            efficiency = self.battery.discharging_efficiency
            target_charging_power = target_energy_change * efficiency / (15. / 60.)

        # actual power is bounded by the properties of the battery
        actual_charging_power = np.clip(target_charging_power,
                                        self.battery.discharging_power_limit,
                                        self.battery.charging_power_limit)

        # actual energy change is based on the actual power possible and the efficiency
        if actual_charging_power >= 0:
            actual_energy_change = actual_charging_power * (15. / 60.) * efficiency
        else:
            actual_energy_change = actual_charging_power * (15. / 60.) / efficiency

        # what we need from the grid = (the power put into the battery + the consumption) - what is available from pv
        grid_energy = (actual_charging_power * (15. / 60.) + actual_consumption) - actual_pv

        # if positive, we are buying from the grid; if negative, we are selling
        return grid_energy, actual_energy_change


start = time.time()
if __name__ == '__main__':
    # simulation_dir = (Path(__file__)/os.pardir/os.pardir).resolve()
    # data_dir = simulation_dir/'data'
    # output_dir = simulation_dir/'output'

    # load available metadata to determine the runs
    metadata_path = 'metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0, sep=';')
    #metadata = metadata[metadata.index == 12]
    batteries = [1, 2]
    # store results of each run
    results = []

    # # execute two runs with each battery for every row in the metadata file: metadata.shape[0]
    for site_id, parameters in tqdm(metadata.iterrows(), desc='sites\t\t\t', total=metadata.shape[0]):
        site_data_path = "submit" + str(site_id) + '.csv'

        # if site_data_path.exists():
        site_data = pd.read_csv(site_data_path, index_col='timestamp', sep=';', parse_dates=['timestamp'])
        # site_data = pd.read_csv(site_data_path,
        #                      parse_dates=['timestamp'],
        #                     index_col='timestamp')

        for batt_id in tqdm(batteries, desc=' > batteries \t\t'):
            # create the battery for this run
            # (Note: Quantities in kW are converted to watts here)
            batt = Battery(capacity=parameters[f"Battery_{batt_id}_Capacity"] * 1000,
                           charging_power_limit=parameters[f"Battery_{batt_id}_Power"] * 1000,
                           discharging_power_limit=-parameters[f"Battery_{batt_id}_Power"] * 1000,
                           charging_efficiency=parameters[f"Battery_{batt_id}_Charge_Efficiency"],
                           discharging_efficiency=parameters[f"Battery_{batt_id}_Discharge_Efficiency"])

            # execute the simulation for each simulation period in the data
            n_periods = site_data.period_id.nunique()
            for g_id, g_df in tqdm(site_data.groupby('period_id'), total=n_periods, desc=' > > periods\t\t'):
                # reset battery to no charge before simulation
                batt.current_charge = 0

                sim = Simulation(g_df, batt, site_id)
                money_spent, money_no_batt = sim.run()

                # store the results
                results.append({
                    'run_id': f"{site_id}_{batt_id}_{g_id}",
                    'site_id': site_id,
                    'battery_id': batt_id,
                    'period_id': g_id,
                    'money_spent': money_spent,
                    'money_no_batt': money_no_batt,
                    'score': (money_spent - money_no_batt) / np.abs(money_no_batt),
                })

    # write all results out to a file
    results_df = pd.DataFrame(results).set_index('run_id')
    results_df = results_df[['site_id', 'battery_id', 'period_id', 'money_spent', 'money_no_batt', 'score']]
    results_df.groupby('period_id').mean('score')
    score = results_df.mean()['score']
    results_df.to_csv('DOCPLEX 5 variables MILP.csv')
    print(score)
end = time.time()
print(end - start)
