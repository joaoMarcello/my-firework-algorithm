from datetime import datetime, timedelta

import numpy as np

from utils import load_data
from fwa import FWA


def fitness(solution):
    total_penalty = 0
    schedule = np.array(solution, dtype=int).reshape((n_employees, n_days))

    # Mapeamento para acesso rápido ao índice do funcionário
    employee_id_to_index = {eid: i for i, eid in enumerate(employees.keys())}

    # Cobertura mínima
    for day_offset in range(n_days):
        date = start + timedelta(days=day_offset)
        day_name = date.strftime("%A")
        shift_counts = {sid: 0 for sid in shifts}
        for emp_id, emp_idx in employee_id_to_index.items():
            assigned_shift = shift_ids[schedule[emp_idx, day_offset]]
            if assigned_shift == "OFF":  # folga não conta para cobertura
                continue
            shift_counts[assigned_shift] += 1

        if day_name in cover:
            for sid, required in cover[day_name].items():
                if shift_counts.get(sid, 0) < required:
                    total_penalty += (required - shift_counts.get(sid, 0)) * 10

    # Pedidos de folga (ShiftOffRequests)
    for req in off_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        if emp_idx is None:
            continue
        day_idx = (datetime.strptime(req["Date"], "%Y-%m-%d") - start).days
        if 0 <= day_idx < n_days:
            assigned_shift = shift_ids[schedule[emp_idx, day_idx]]
            if assigned_shift != "OFF":  # penaliza se folga não foi atendida
                total_penalty += req["Weight"]

    # Pedidos de turno desejado (ShiftOnRequests)
    for req in on_reqs:
        emp_idx = employee_id_to_index.get(req["EmployeeID"])
        if emp_idx is None:
            continue
        day_idx = (datetime.strptime(req["Date"], "%Y-%m-%d") - start).days
        if 0 <= day_idx < n_days:
            assigned_shift = shift_ids[schedule[emp_idx, day_idx]]
            if assigned_shift != req["ShiftTypeID"]:
                total_penalty += req["Weight"]

    return total_penalty


if __name__ == '__main__':
    data = load_data(xml_path='data/ORTEC01.xml')
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
    shift_ids = list(shifts.keys())
    n_shift_types = len(shift_ids)

    solution_size = n_employees * n_days
    bounds = [(0, n_shift_types - 1)] * solution_size  # cada valor representa um turno possível

    fwa = FWA(func=fitness, dim=solution_size, bounds=bounds)
    fwa.load_prob(n=5, m=50, max_iter=100)
    fwa.run()

    print("Melhor valor encontrado:", fwa.best_value)

    # Decodificando para visualização
    schedule = np.array(fwa.best_solution, dtype=int).reshape((n_employees, n_days))
    for emp_idx, emp_id in enumerate(employees):
        line = [shift_ids[s] for s in schedule[emp_idx]]
        print(f"{employees[emp_id]['Name']}: {line}")






