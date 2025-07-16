import argparse
from datetime import datetime, timedelta, time
import random

import numpy as np

from fwa import FWA
from utils import load_data
from constraints.hard_constraints import *

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def fitness(solution):
    schedule = np.rint(solution).astype(int).reshape((n_employees, n_days))
    total_penalty = 0

    decoded = decode_solution(
        solution,
        employees,
        start_date,
        n_days,
        shift_ids
    )
    # Penalidade por pedidos de folga não atendidos
    off_idx = shift_id_to_index["OFF"]
    for req in off_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        day_idx = (req["Date"] - start_date).days
        if emp_idx is not None and 0 <= day_idx < n_days:
            if schedule[emp_idx, day_idx] != off_idx:
                total_penalty += req["Weight"]

    # Penalidade por turnos desejados não atendidos
    for req in on_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        day_idx = (req["Date"] - start_date).days
        requested_idx = shift_id_to_index.get(req["ShiftTypeID"])
        if emp_idx is not None and 0 <= day_idx < n_days and requested_idx is not None:
            if schedule[emp_idx, day_idx] != requested_idx:
                total_penalty += req["Weight"]

    # Penalidade por exceder horas do contrato (+4h tolerância)
    for emp_id, emp_idx in employee_id_to_index.items():
        contract_id = employees[emp_id]["ContractID"]
        contract_rules = contracts.get(contract_id, [])
        max_hours = 0

        for rule in contract_rules:
            if "Max 36 hours" in str(rule.get("Max", {}).get("Label", "")):
                max_hours = rule["Max"]["Count"]
                break

        total_hours = 0.0
        for d in range(n_days):
            shift_idx = schedule[emp_idx, d]
            shift_id = shift_ids[shift_idx]
            duration = float(shifts[shift_id].get("Duration", 0))
            total_hours += duration

        if max_hours > 0 and total_hours > max_hours + 4:
            excess = total_hours - (max_hours + 4)
            total_penalty += int(excess) * 10

    # Penalidades de padrões contratuais e turnos por semana
    for emp_id, emp_idx in employee_id_to_index.items():
        contract_id = employees[emp_id]["ContractID"]
        shift_labels = [shift_ids[s] for s in schedule[emp_idx]]
        rules = contracts.get(contract_id, [])

        for rule in rules:
            pattern = rule.get("Pattern", [])
            label = str(rule.get("Max", {}).get("Label", ""))
            max_count = rule.get("Max", {}).get("Count", None)
            weight = rule.get("Max", {}).get("Weight", 0)

            if max_count is None:
                continue

            # Turnos noturnos
            if "Max 3 nights" in label:
                night_count = shift_labels.count("N")
                if night_count > max_count:
                    total_penalty += (night_count - max_count) * weight

            # Fins de semana trabalhados
            elif "Max 3 working weekends" in label:
                worked_weekends = 0
                for week in range(0, n_days, 7):
                    week_shifts = shift_labels[week:week+7]
                    if any(shift != "OFF" for i, shift in enumerate(week_shifts) if (start_date + timedelta(days=week+i)).weekday() in (5, 6)):
                        worked_weekends += 1
                if worked_weekends > max_count:
                    total_penalty += (worked_weekends - max_count) * weight

            # Dias de trabalho por semana (min/max)
            elif "Shifts per week" in label:
                region_start = rule.get("RegionStart", 0)
                region_end = rule.get("RegionEnd", n_days - 1)

                min_dict = rule.get("Min")
                max_dict = rule.get("Max")

                min_count = min_dict.get("Count") if min_dict else None
                max_count = max_dict.get("Count") if max_dict else None
                weight = max_dict.get("Weight") if max_dict else 0

                for w_start in range(region_start, region_end + 1, 7):
                    w_end = min(w_start + 6, region_end)
                    work_days = sum(1 for s in shift_labels[w_start:w_end+1] if s != "OFF")
                    if min_count and work_days < min_count:
                        total_penalty += ((min_count - work_days) ** 2) * weight
                    if max_count and work_days > max_count:
                        total_penalty += ((work_days - max_count) ** 2) * weight

            # Padrões indesejados (ex: N, N, D)
            if pattern:
                pattern_len = len(pattern)
                for i in range(len(shift_labels) - pattern_len + 1):
                    match = True
                    for j in range(pattern_len):
                        pat_el = pattern[j]
                        shift_at_day = shift_labels[i + j]

                        if "Shift" in pat_el and pat_el["Shift"] != "-" and shift_at_day != pat_el["Shift"]:
                            match = False
                        elif "NotShift" in pat_el and shift_at_day == pat_el["NotShift"]:
                            match = False
                        elif "ShiftGroup" in pat_el:
                            if shift_at_day not in shift_groups.get(pat_el["ShiftGroup"], []):
                                match = False

                    if match:
                        total_penalty += weight

    total_penalty += hard_cover_fulfillment(decoded, cover, start_date, end_date)
    total_penalty += hard_exceed_contract_hours(decoded, shifts, employees, contracts)

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

    fwa = FWA(func=fitness, 
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
               max_iter=args.fwa_max_iter)
    
    fwa.set_problem_context(start_date=start_date,
                            n_days=n_days,
                            n_employees=n_employees,
                            shift_id_to_index=shift_id_to_index,
                            shift_ids=shift_ids)

    print(n_employees)    
    fwa.run()
    fwa.save_to_disc(path=args.save_file + '.json')

    print("Melhor valor encontrado:", fwa.best_value)


    fwa.plot_history_from_file(json_path=args.save_file + '.json', save_path=args.save_file + '.png')

    # # Decodificando para visualização
    # schedule = np.array(fwa.best_solution, dtype=int).reshape((n_employees, n_days))
    # for emp_idx, emp_id in enumerate(employees):
    #     line = [shift_ids[s] for s in schedule[emp_idx]]
    #     print(f"{employees[emp_id]['Name']}: {line}")






