import argparse
from datetime import datetime, timedelta
import random

import numpy as np

from fwa import FWA
from utils import load_data

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def contract_penalty(emp_id, emp_schedule):
    total = 0
    contract_id = employees[emp_id]["ContractID"]
    shift_labels = [shift_ids[idx] for idx in emp_schedule]
    rules = contracts.get(contract_id, [])

    for rule in rules:
        pattern = rule["Pattern"]
        label = rule.get("Label", "")
        max_count = int(rule.get("Max", 999999)) if rule.get("Max") else None
        weight = int(rule.get("Weight", 0))

        # Max total shifts
        if "Max 20 shifts" in label:
            worked_days = sum(1 for s in shift_labels if s != "OFF")
            if worked_days > max_count:
                total += (worked_days - max_count) * weight

        # Max night shifts
        elif "Max 3 nights" in label:
            night_shift = "N"
            night_count = shift_labels.count(night_shift)
            if night_count > max_count:
                total += (night_count - max_count) * weight

        # Min 2 consecutive free days
        elif "Min 2 consecutive free days" in label:
            consec_frees = 0
            max_consec_frees = 0
            for shift in shift_labels:
                if shift == "OFF":
                    consec_frees += 1
                    max_consec_frees = max(max_consec_frees, consec_frees)
                else:
                    consec_frees = 0
            if max_consec_frees < 2:
                total += weight

        # Max working weekends
        elif "Max 3 working weekends" in label:
            weekends_worked = 0
            for i in range(n_days):
                date = start + timedelta(days=i)
                if date.weekday() in (5, 6):  # Sábado ou domingo
                    if shift_labels[i] != "OFF":
                        weekends_worked += 1
                        break
            if weekends_worked > max_count:
                total += (weekends_worked - max_count) * weight

        # Shifts per week (com RegionStart e RegionEnd)
        elif "Shifts per week" in label:
            region_start = int(rule.get("RegionStart", 0))
            region_end = int(rule.get("RegionEnd", n_days - 1))
            region_shifts = shift_labels[region_start:region_end + 1]
            worked = sum(1 for s in region_shifts if s != "OFF")
            min_count = int(rule.get("Min", 0)) if rule.get("Min") else None
            max_count = int(rule.get("Max", 999)) if rule.get("Max") else None
            weight = int(rule.get("Weight", 0))

            if min_count and worked < min_count:
                penalty = (min_count - worked) ** 2 * weight
                total += penalty
            if max_count and worked > max_count:
                penalty = (worked - max_count) ** 2 * weight
                total += penalty

    return total

def fitness(solution):
    total_penalty = 0
    # schedule = np.array(solution, dtype=int).reshape((n_employees, n_days))
    schedule = np.rint(solution).astype(int).reshape((n_employees, n_days))

    # Cobertura mínima
    for day_offset in range(n_days):
        date = start + timedelta(days=day_offset)
        day_name = date.strftime("%A")
        shift_counts = {sid: 0 for sid in shifts}
        for emp_id, emp_idx in employee_id_to_index.items():
            assigned_shift_id = shift_ids[schedule[emp_idx, day_offset]]  # ID do turno
            if assigned_shift_id == "OFF":  # folga não conta para cobertura
                continue
            shift_counts[assigned_shift_id] += 1

        if day_name in cover:
            for sid, required in cover[day_name].items():
                if shift_counts.get(sid, 0) < required:
                    total_penalty += (required - shift_counts.get(sid, 0)) * 10

    # Pedidos de folga (ShiftOffRequests)
    off_idx = shift_id_to_index["OFF"]  # índice do turno OFF para comparar
    for req in off_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        if emp_idx is None:
            continue
        day_idx = (datetime.strptime(req["Date"], "%Y-%m-%d") - start).days
        if 0 <= day_idx < n_days:
            assigned_idx = schedule[emp_idx, day_idx]  # índice do turno atribuído
            if assigned_idx != off_idx:  # penaliza se folga não foi atendida
                total_penalty += req["Weight"]

    # Pedidos de turno desejado (ShiftOnRequests)
    for req in on_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        if emp_idx is None:
            continue
        day_idx = (datetime.strptime(req["Date"], "%Y-%m-%d") - start).days
        if 0 <= day_idx < n_days:
            assigned_idx = schedule[emp_idx, day_idx]
            requested_idx = shift_id_to_index.get(req["ShiftTypeID"], None)
            if requested_idx is None:
                # Pode ser que tenha um turno no pedido que não está no conjunto, tratar caso necessário
                continue
            if assigned_idx != requested_idx:
                total_penalty += req["Weight"]

    # Penalidades contratuais por funcionário
    for emp_id, emp_idx in employee_id_to_index.items():
        emp_schedule = schedule[emp_idx]
        total_penalty += contract_penalty(emp_id, emp_schedule)

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
    shifts = data['shifts']
    employees = data['employees']
    contracts = data['contracts']
    cover = data['cover_requirements']
    off_reqs = data['shift_off_requests']
    on_reqs = data['shift_on_requests']
    start_date = data['start_date']
    end_date = data['end_date']

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    n_days = (end - start).days + 1
    n_employees = len(employees)
    shift_ids = sorted(shifts.keys())
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
    
    fwa.run()
    fwa.save_to_disc(path=args.save_file + '.json')

    print("Melhor valor encontrado:", fwa.best_value)

    fwa.plot_history_from_file(json_path=args.save_file + '.json', save_path=args.save_file + '.png')

    # # Decodificando para visualização
    # schedule = np.array(fwa.best_solution, dtype=int).reshape((n_employees, n_days))
    # for emp_idx, emp_id in enumerate(employees):
    #     line = [shift_ids[s] for s in schedule[emp_idx]]
    #     print(f"{employees[emp_id]['Name']}: {line}")






