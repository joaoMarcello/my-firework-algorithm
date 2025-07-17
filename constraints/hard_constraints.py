import numpy as np
from datetime import timedelta





def decode_solution(solution, employees, start_date, n_days, shift_ids):
    n_employees = len(employees)
    employee_ids = list(employees.keys())

    # Aplica o mesmo processamento do seu código principal
    schedule = np.rint(solution).astype(int).reshape((n_employees, n_days))

    decoded = {}
    for emp_idx, emp_id in enumerate(employee_ids):
        decoded[emp_id] = {}
        for day_idx in range(n_days):
            date = start_date + timedelta(days=day_idx)
            shift_index = schedule[emp_idx, day_idx]
            shift_index = max(0, min(shift_index, len(shift_ids) - 1))  # segurança de índice
            decoded[emp_id][date] = shift_ids[shift_index]
    return decoded


# 1. Cobertura obrigatória
def hard_cover_fulfillment(decoded_solution, cover_requirements, start_date, end_date):
    from collections import Counter
    from datetime import timedelta

    penalty = 0
    current_date = start_date
    while current_date <= end_date:
        weekday = current_date.strftime('%A')
        required = cover_requirements.get(weekday, {})
        assigned = Counter()

        for schedule in decoded_solution.values():
            shift_id = schedule.get(current_date)
            if shift_id != "OFF":
                assigned[shift_id] += 1

        for shift_id, demand in required.items():
            if assigned[shift_id] < demand:
                penalty += (demand - assigned[shift_id]) * 1000

        current_date += timedelta(days=1)
    return penalty


# Limite de horas +4h do contrato
def hard_exceed_contract_hours(decoded_solution, shifts, employees, contracts):
    penalty = 0
    for emp_id, schedule in decoded_solution.items():
        contract_id = employees[emp_id]["ContractID"]
        contract_rules = contracts[contract_id]
        max_minutes = 0

        for rule in contract_rules:
            if rule["Max"]["Label"] == "TotalMinutes":
                max_minutes = rule["Max"]["Count"]
                break

        total_hours = sum(shifts[shift]["Duration"] for shift in schedule.values() if shift != "OFF")
        allowed_hours = (max_minutes / 60.0) + 4
        if total_hours > allowed_hours:
            penalty += int((total_hours - allowed_hours) * 100)
    return penalty
"""
4. A média de 36h semanais (13 semanas sem noite)
Como você ainda não lida com semanas/13 semanas/
turnos noturnos explicitamente, vamos deixar isso 
para depois ou definir turnos noturnos via
Exemplo: se o turno começa após 22h ou termina antes de 6h
"""

def is_night_shift(shift):
    start = shift["StartTime"]
    end = shift["EndTime"]
    if not start or not end:
        return False
    h1 = int(start.split(":")[0])
    h2 = int(end.split(":")[0])
    return h1 >= 22 or h2 <= 6
