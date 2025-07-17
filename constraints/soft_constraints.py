import numpy as np
from datetime import timedelta

def penalize_min_consecutive_free_days_all(schedule, shift_index_for_off, employees, contracts, default_weight=100):
    """
    Penaliza todos os enfermeiros que não possuem pelo menos 2 dias consecutivos de folga, conforme regras do contrato.

    Parâmetros:
        schedule: ndarray (n_employees, n_days), com índices dos turnos
        shift_index_for_off: índice inteiro representando folga (ex: '-' ou 'OFF')
        employees: dict com dados dos enfermeiros (ID -> {..., 'ContractID': ...})
        contracts: dict com dados dos contratos (ID -> regras com labels como 'Min 2 consecutive free days')
        default_weight: valor padrão caso a regra não tenha Weight definido

    Retorna:
        penalty: soma das penalidades para todos os enfermeiros
    """
    penalty = 0
    n_employees = schedule.shape[0]

    for nurse_index in range(n_employees):
        # Identifica contrato do enfermeiro
        emp_id = list(employees.keys())[nurse_index]
        contract_id = employees[emp_id]["ContractID"]

        # Procura a regra "Min 2 consecutive free days" para o contrato
        contract_rules = contracts.get(contract_id, [])
        rule = None
        for r in contract_rules:
            label = r.get("Max", {}).get("Label", "") or r.get("Min", {}).get("Label", "")
            if label and "min 2 consecutive free days" in label.lower():
                rule = r
                break

        if not rule:
            continue  # Nenhuma regra correspondente no contrato

        expected_min_consecutive = 2
        rule_weight = rule.get("Max", {}).get("Weight", default_weight)

        shifts = schedule[nurse_index]
        n_days = len(shifts)

        # Verifica se há pelo menos 2 dias consecutivos de folga
        has_consecutive_off = any(
            np.all(shifts[i:i+expected_min_consecutive] == shift_index_for_off)
            for i in range(n_days - expected_min_consecutive + 1)
        )

        if not has_consecutive_off:
            penalty += rule_weight

    return int(penalty)


def penalize_max_nights(schedule: np.ndarray, shift_index_for_night: int, max_nights: int, weight: float):
    """
    Penaliza enfermeiros que ultrapassem o máximo de turnos noturnos permitidos.

    Args:
        schedule (np.ndarray): Matriz de shape (n_employees, n_days) com índices dos turnos.
        shift_index_for_night (int): Índice do turno "N" (noite) no vetor de turnos.
        max_nights (int): Número máximo permitido de turnos noturnos.
        weight (float): Penalidade aplicada por violação.

    Returns:
        float: Penalidade total.
    """
    penalty = 0.0
    n_employees, n_days = schedule.shape

    for nurse_idx in range(n_employees):
        nurse_shifts = schedule[nurse_idx]
        nights_count = np.sum(nurse_shifts == shift_index_for_night)

        if nights_count > max_nights:
            penalty += weight

    return penalty


def penalize_max_nights_all_nurses(schedule, employees, contracts, shift_id_to_index):
    """
    Penaliza os enfermeiros que excedem o máximo permitido de turnos 'N' (noite) conforme seu contrato.
    
    schedule: numpy array shape (n_employees, n_days), com índices de turno.
    employees: dict employee_id -> dict com pelo menos 'ContractID'.
    contracts: dict contract_id -> lista de regras (dict).
    shift_id_to_index: dict shift_id (ex: 'N', 'D', '-') -> índice no schedule.
    
    Retorna: penalidade total (float).
    """
    penalty = 0
    n_employees, n_days = schedule.shape
    
    # índice do turno 'N' (noite)
    night_shift_index = shift_id_to_index.get('N')
    if night_shift_index is None:
        # Se não existe turno 'N', não penaliza
        return 0
    
    # Obtem lista de employee ids na ordem do schedule (assumindo ordem consistente)
    employee_ids = list(employees.keys())
    
    for nurse_idx in range(n_employees):
        emp_id = employee_ids[nurse_idx]
        contract_id = employees[emp_id].get("ContractID")
        
        if contract_id not in contracts:
            # Se não tem contrato válido, ignora
            continue
        
        rules = contracts[contract_id]  # lista de regras
        
        # Para essa regra, procura "Max" que limita turnos 'N'
        for rule in rules:
            max_rule = rule.get("Max")
            patterns = rule.get("Pattern", [])
            
            if max_rule is None or not patterns:
                continue
            
            max_count = max_rule.get("Count")
            max_weight = max_rule.get("Weight", 0)
            
            # Ignora regras com max_count 0 (normalmente significa sem limite)
            if max_count == 0:
                continue
            
            # Verifica se o padrão envolve turnos 'N'
            # Pode haver múltiplos patterns, checamos se algum tem Shift == 'N'
            matches_night = any(
                (pat.get("Shift") == "N") or (pat.get("ShiftGroup") == "N")
                for pat in patterns
            )
            if not matches_night:
                continue
            
            # Conta quantos turnos 'N' o enfermeiro tem na escala
            nurse_shifts = schedule[nurse_idx]
            night_shifts_count = np.sum(nurse_shifts == night_shift_index)
            
            if night_shifts_count > max_count:
                penalty += max_weight
    
    return int(penalty)


def penalize_max_working_weekends(schedule, contracts, employees, shift_id_to_index, start_date):
    """
    Penaliza enfermeiros que trabalhem em mais de max_allowed finais de semana (noite).
    Considera turno 'N' como turno noturno.
    
    Args:
        schedule: numpy array (n_employees, n_days), com índices dos turnos.
        contracts: dict de contratos carregados pelo load_data.
        employees: dict com info dos empregados, inclusive o contrato.
        shift_id_to_index: dict para mapear código do turno para índice na schedule.
        start_date: datetime.date da data inicial da escala.
    
    Returns:
        penalty: float, soma das penalidades por excesso de finais de semana trabalhados.
    """

    n_employees, n_days = schedule.shape
    penalty = 0

    # Índice do turno noturno
    night_shift_idx = shift_id_to_index.get('N')
    if night_shift_idx is None:
        # Se não há turno N, regra não aplicável
        return 0

    # Para calcular os finais de semana, mapeamos os dias para dias da semana
    # Segunda=0, ..., Domingo=6
    day_indices = np.array([(start_date + timedelta(days=i)).weekday() for i in range(n_days)])

    for emp_idx, (emp_id, emp_info) in enumerate(employees.items()):
        contract_id = emp_info.get("ContractID")
        if contract_id is None or contract_id not in contracts:
            continue  # Sem contrato válido, pula

        contract_rules = contracts[contract_id]

        # Busca regra "Max 3 working weekends"
        max_weekends_rule = None
        for rule in contract_rules:
            label = rule.get("Max", {}).get("Label", "")
            if "Max" in rule and label and "working weekends" in label.lower():
                max_weekends_rule = rule["Max"]
                break

        if not max_weekends_rule:
            continue  # Regra não encontrada, pula

        max_allowed = max_weekends_rule.get("Count", 3)
        weight = max_weekends_rule.get("Weight", 1000)

        # Conta quantos finais de semana o enfermeiro trabalhou
        # Para isso, pega dias de sábado/domingo e verifica se há turno N em algum desses dias
        emp_schedule = schedule[emp_idx]

        # Armazena se trabalhou no final de semana i
        worked_weekends = 0

        # Vamos identificar os finais de semana no horizonte:
        # Procura pares consecutivos sábado (5) e domingo (6)
        weekend_starts = [i for i, d in enumerate(day_indices[:-1]) if d == 5 and i+1 < n_days and day_indices[i+1] == 6]

        for start_day in weekend_starts:
            saturday_shift = emp_schedule[start_day]
            sunday_shift = emp_schedule[start_day + 1]

            # Trabalhou se em sábado OU domingo tiver turno N
            if saturday_shift == night_shift_idx or sunday_shift == night_shift_idx:
                worked_weekends += 1

        # # Penaliza se passou do limite
        # if worked_weekends > max_allowed:
        #     penalty += weight

        excess = worked_weekends - max_allowed
        if excess > 0:
            penalty += excess * weight


    return penalty


def penalize_shift_off_requests(schedule, shift_off_requests, employees, shift_id_to_index, start_date):
    """
    Penaliza alocações que violam pedidos de folga (ShiftOffRequests).

    Args:
        schedule: numpy array (n_employees, n_days) com índices de turnos.
        shift_off_requests: lista de dicts com chaves "EmployeeID", "ShiftTypeID", "Date", "Weight".
        employees: dict com info dos empregados, incluindo a ordem.
        shift_id_to_index: dict do ID do turno para índice na schedule.
        start_date: datetime.date inicial da escala.

    Returns:
        penalty: float, soma das penalidades por violar pedidos de folga.
    """
    penalty = 0
    # Criar mapeamento de EmployeeID para índice no schedule
    emp_id_to_idx = {emp_id: idx for idx, emp_id in enumerate(employees.keys())}

    for request in shift_off_requests:
        emp_id = request["EmployeeID"]
        shift_id = request["ShiftTypeID"]
        date = request["Date"]
        weight = request["Weight"]

        if emp_id not in emp_id_to_idx:
            continue  # Ignora se funcionário não estiver na escala

        emp_idx = emp_id_to_idx[emp_id]
        day_idx = (date - start_date).days

        if day_idx < 0 or day_idx >= schedule.shape[1]:
            continue  # Data fora do horizonte da escala

        # Verifica se o turno alocado é o turno pedido como folga
        shift_idx = shift_id_to_index.get(shift_id)
        if shift_idx is None:
            continue  # Turno não encontrado, ignora

        if schedule[emp_idx, day_idx] == shift_idx:
            penalty += weight

    return penalty


def penalize_shift_on_requests(schedule, shift_on_requests, employees, shift_id_to_index, start_date):
    """
    Penaliza quando um enfermeiro NÃO é escalado para o turno solicitado em determinada data.
    
    Args:
        schedule: numpy array (n_employees, n_days), com índices dos turnos.
        shift_on_requests: lista de dicts, cada um com keys: EmployeeID, ShiftTypeID, Date, Weight.
        employees: dict com info dos empregados, incluindo a ordem deles na schedule (chave deve bater).
        shift_id_to_index: dict para mapear código do turno para índice na schedule.
        start_date: datetime.date do início da escala.
        
    Returns:
        penalty: float, soma das penalidades pelas solicitações não atendidas.
    """

    penalty = 0

    # Mapear EmployeeID para índice na schedule para acesso rápido
    emp_id_to_idx = {emp_id: idx for idx, emp_id in enumerate(employees.keys())}

    for req in shift_on_requests:
        emp_id = req["EmployeeID"]
        requested_shift_id = req["ShiftTypeID"]
        date = req["Date"]
        weight = req.get("Weight", 1)

        # Se enfermeiro não estiver na escala, ignore
        if emp_id not in emp_id_to_idx:
            continue

        emp_idx = emp_id_to_idx[emp_id]
        day_idx = (date - start_date).days

        # Se a data estiver fora do horizonte da escala, ignore
        if day_idx < 0 or day_idx >= schedule.shape[1]:
            continue

        requested_shift_idx = shift_id_to_index.get(requested_shift_id)

        # Se não encontrar o índice do turno solicitado, ignore
        if requested_shift_idx is None:
            continue

        # Verifica se o enfermeiro está escalado para o turno pedido nesse dia
        actual_shift_idx = schedule[emp_idx, day_idx]

        if actual_shift_idx != requested_shift_idx:
            penalty += weight

    return penalty



