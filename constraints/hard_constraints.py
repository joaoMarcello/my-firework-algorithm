from datetime import timedelta
import re

import numpy as np

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
def hard_cover_fulfillment(schedule, cover_requirements, start_date, shift_id_to_index, cover_weights, n_days):
    penalty = 0

    # Para cada dia da janela
    for day_offset in range(n_days):
        current_date = start_date + timedelta(days=day_offset)
        day_name = current_date.strftime("%A")  # "Monday", "Tuesday", etc
        
        # Cobertura exigida para esse dia da semana
        day_cover = cover_requirements.get(day_name, {})
        
        # Contagem de enfermeiros escalados por turno
        # Inicializa contagem zerada para cada turno
        scheduled_counts = {shift_id: 0 for shift_id in day_cover.keys()}
        
        for emp_idx in range(schedule.shape[0]):
            shift_idx = schedule[emp_idx, day_offset]
            # Pega o shift_id a partir do índice do turno
            shift_id = None
            for sid, idx in shift_id_to_index.items():
                if idx == shift_idx:
                    shift_id = sid
                    break

            if shift_id in scheduled_counts:
                scheduled_counts[shift_id] += 1

        # Calcula penalidades para esse dia
        for shift_id, required in day_cover.items():
            scheduled = scheduled_counts.get(shift_id, 0)
            if scheduled < required:
                diff = required - scheduled
                # penalty += diff * cover_weights.get("PrefUnderStaffing", 10000)
                penalty += cover_weights.get("PrefUnderStaffing", 10000)
            elif scheduled > required:
                diff = scheduled - required
                # penalty += diff * cover_weights.get("PrefOverStaffing", 10000)
                penalty += cover_weights.get("PrefOverStaffing", 10000)

    return penalty


# # Checa se extrapolou a quantidade de turnos (pega a informação do contrato)
# def hard_max_shifts_from_contract(decoded_solution, employees, contracts):
#     penalty = 0

#     for emp_id, schedule in decoded_solution.items():
#         contract_id = employees[emp_id]["ContractID"]
#         contract_rules = contracts[contract_id]

#         for rule in contract_rules:
#             label = str(rule.get("Max", {}).get("Label", ""))
#             match = re.match(r"Max (\d+) shifts", label)

#             if match:
#                 max_shifts = int(rule["Max"]["Count"])  # ou: int(match.group(1))
#                 weight = int(rule["Max"]["Weight"])

#                 # Conta turnos (exceto "OFF") no mês
#                 worked_days = sum(1 for shift in schedule.values() if shift != "OFF")

#                 if worked_days > max_shifts:
#                     excess = worked_days - max_shifts
#                     penalty += excess * weight

#     return penalty

# Checa se extrapolou a quantidade de turnos (pega a informação do contrato)
def hard_max_shifts_from_contract_matrix(schedule_matrix, employees, contracts, employee_id_to_index, shift_off_index):
    penalty = 0

    for emp_id, emp_data in employees.items():
        emp_idx = employee_id_to_index[emp_id]
        contract_id = emp_data["ContractID"]
        contract_rules = contracts[contract_id]

        emp_schedule = schedule_matrix[emp_idx]
        worked_days = np.sum(emp_schedule != shift_off_index)

        for rule in contract_rules:
            label = str(rule.get("Max", {}).get("Label", ""))
            match = re.match(r"Max (\d+) shifts", label)

            if match:
                max_shifts = int(rule["Max"]["Count"])  # ou: int(match.group(1))
                weight = int(rule["Max"]["Weight"])
                
                if worked_days > max_shifts:
                    excess = worked_days - max_shifts
                    penalty += excess * weight

    return int(penalty)

# Checa a regra do contrato Shifts per week
# a quantidade de turno
def hard_check_bounded_shifts_in_region(
    schedule_matrix,                
    employees,                      
    contracts,                      
    employee_id_to_index,           
    shift_off_index,                
    n_days                         
):
    penalty = 0

    bounded_shift_rules = {
        cid: [
            rule for rule in rules
            if "Pattern" in rule and "Max" in rule and rule["Max"].get("Label") == "Shifts per week"
        ]
        for cid, rules in contracts.items()
    }

    for emp_id, emp_data in employees.items():
        row = employee_id_to_index[emp_id]
        emp_schedule = schedule_matrix[row]
        contract_id = emp_data["ContractID"]
        rules = bounded_shift_rules.get(contract_id)

        if not rules:
            continue

        for rule in rules:
            region_start = int(rule.get("RegionStart", 0))
            region_end = int(rule.get("RegionEnd", n_days - 1))
            region_end = min(region_end, n_days - 1)

            sub_schedule = emp_schedule[region_start:region_end + 1]
            worked_days = np.sum(sub_schedule != shift_off_index)

            # Penalidade Min
            if "Min" in rule and rule["Min"] is not None:
                min_count = int(rule["Min"]["Count"])
                min_weight = int(rule["Min"]["Weight"])
                min_func = rule["Min"].get("Function", "Linear")
                diff = max(0, min_count - worked_days)
                penalty += min_weight * (diff ** 2 if min_func == "Quadratic" else diff)

            # Penalidade Max
            if "Max" in rule and rule["Max"] is not None:
                max_count = int(rule["Max"]["Count"])
                max_weight = int(rule["Max"]["Weight"])
                max_func = rule["Max"].get("Function", "Linear")
                diff = max(0, worked_days - max_count)
                penalty += max_weight * (diff ** 2 if max_func == "Quadratic" else diff)

    return int(penalty)


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
