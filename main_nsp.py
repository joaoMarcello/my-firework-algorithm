import argparse
from datetime import datetime, timedelta, time
import random

import numpy as np

from fwa import FWA
from fwa_discrete import DiscreteFWA
from utils import load_data, convert_schedule_to_str
from constraints.hard_constraints import *
from constraints.soft_constraints import *

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def fitness(solution):
    schedule = np.rint(solution).astype(int).reshape((n_employees, n_days))
    schedule_str_list = convert_schedule_to_str(schedule, shift_ids)
    schedule_str = ''.join(schedule_str_list)

    total_penalty = 0

    #===================================================================
    total_penalty += penalize_pattern(schedule_str, 'NE', 
    penalty_per_occurrence=10)

    total_penalty += penalize_late_shift_series(schedule_str_list)
    total_penalty += penalize_early_shift_series(schedule_str_list)

    total_penalty += penalize_insufficient_rest(schedule_str_list)
    total_penalty += penalize_standalone_shifts(schedule_str_list)

    total_penalty += penalize_weekend_pattern(schedule, start_date, shift_id_to_index)
    #==================================================================

    total_penalty += penalize_min_consecutive_free_days_all(schedule, shift_id_to_index["OFF"], employees, contracts)

    total_penalty += penalize_max_nights_all_nurses(schedule, employees, contracts, shift_id_to_index)

    total_penalty += penalize_max_working_weekends(schedule, contracts, employees, shift_id_to_index, start_date)

    total_penalty += penalize_shift_off_requests(schedule, off_reqs, employees, shift_id_to_index, start_date)

    total_penalty += penalize_shift_on_requests(schedule, on_reqs, employees, shift_id_to_index, start_date)

    total_penalty += hard_cover_fulfillment(schedule, cover, start_date, shift_id_to_index, cover_weights, n_days)

    total_penalty += hard_max_shifts_from_contract(schedule, employees, contracts, employee_id_to_index, shift_off_index=shift_id_to_index["OFF"])

    total_penalty += hard_check_bounded_shifts_in_region(
        schedule_matrix=schedule,
        employees=employees,
        contracts=contracts,
        employee_id_to_index=employee_id_to_index,
        shift_off_index=shift_id_to_index["OFF"],
        n_days=n_days,
    )


    return total_penalty


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FWA for Nurse Scheduling Problem.')
    parser.add_argument('--xml_path', type=str, default='data/ORTEC01.xml')
    parser.add_argument('--save_file', type=str, default='fwa_nsp_result')

    # params for FWA
    # quantidade de fogos
    parser.add_argument('--fwa_n', type=int, default=5)

    # quantidade de faíscas
    parser.add_argument('--fwa_m', type=int, default=50)

    parser.add_argument('--fwa_a', type=float, default=0.04)
    parser.add_argument('--fwa_b', type=float, default=0.8)

    # limite máximo para a amplitude dos fogos
    parser.add_argument('--fwa_a_hat', type=float, default=40)

    # número de faíscas gaussianas
    parser.add_argument('--fwa_m_hat', type=int, default=5)

    # params para joy factor
    parser.add_argument('--fwa_j', type=float, default=0.2)
    parser.add_argument('--fwa_j_hat', type=float, default=0.5)

    # número de iterações do algoritmo
    parser.add_argument('--fwa_max_iter', type=int, default=100)

    # modo de seleção das melhores faíscas para a próxima geração
    parser.add_argument('--fwa_select_mode', type=str, default='distance')

    args = parser.parse_args()

    data = load_data(xml_path=args.xml_path)
    
    def pad_time_string(tstr):
        parts = tstr.split(":")
        parts = [part.zfill(2) for part in parts]
        while len(parts) < 3:
            parts.append("00")
        return ":".join(parts)
    
    shifts = data['shifts']
    shifts = dict(sorted(shifts.items(),
        key=lambda item: datetime.strptime(pad_time_string(item[1]["StartTime"]), "%H:%M:%S").time()
        if item[1]["StartTime"] else time.max))


    shift_groups = data['shift_groups']
    employees = data['employees']
    contracts = data['contracts']
    cover = data['cover_requirements']
    off_reqs = data['shift_off_requests']
    on_reqs = data['shift_on_requests']
    start_date = data['start_date']  # datetime.date já
    end_date = data['end_date']      # datetime.date já
    cover_weights = data['cover_weights']

    n_days = (end_date - start_date).days + 1
    n_employees = len(employees)
    shift_ids = list(shifts.keys())
    n_shift_types = len(shift_ids)

    # Mapeamento para acesso rápido ao índice do funcionário
    employee_id_to_index = {eid: i for i, eid in enumerate(employees.keys())}
    # Mapeamento para acesso rápido ao índice do turno
    shift_id_to_index = {sid: i for i, sid in enumerate(shift_ids)}

    solution_size = n_employees * n_days
    bounds = [(0 - 0.5, n_shift_types - 1 + 0.5)] * solution_size  # cada valor representa um turno possível

    fwa = DiscreteFWA(func=fitness, 
              dim=solution_size, 
              bounds=bounds, 
              selection_method=args.fwa_select_mode,
              seed=SEED)

    fwa.config(n=args.fwa_n, 
               m=args.fwa_m, 
               a = args.fwa_a,
               b = args.fwa_b,
               A_hat = args.fwa_a_hat,
               m_hat= args.fwa_m_hat,
               max_iter=args.fwa_max_iter,
               J=args.fwa_j,
               J_hat=args.fwa_j_hat)
    
    fwa.set_problem_context(start_date=start_date,
                            n_days=n_days,
                            n_employees=n_employees,
                            shift_id_to_index=shift_id_to_index,
                            shift_ids=shift_ids,
                            shift_on_request=on_reqs,
                            employee_id_to_index=employee_id_to_index,
                            cover_requirements=cover)

    fwa.run(time_limit_minutes=60*2)
    fwa.save_to_disc(path=args.save_file + '.json')

    print("Melhor valor encontrado:", fwa.best_value)

    fwa.plot_history_from_file(json_path=args.save_file + '.json', save_path=args.save_file + '.png')

    fwa.save_excel(filename=args.save_file + '.xlsx')

    # fwa.load_best("results_tests_17-07_discrete/fwa_nsp_run_04_for_real.json")
    # fwa.save_excel()

    # # Decodificando para visualização
    # schedule = np.array(fwa.best_solution, dtype=int).reshape((n_employees, n_days))
    # for emp_idx, emp_id in enumerate(employees):
    #     line = [shift_ids[s] for s in schedule[emp_idx]]
    #     print(f"{employees[emp_id]['Name']}: {line}")






