import itertools

def expand_pattern(pattern_elements, shift_groups):
    """
    Expande um padrão ORTEC com elementos <Shift> e <ShiftGroup> em todas as combinações possíveis.
    Retorna uma lista de strings com os padrões expandidos, usando '-' para 'OFF'.
    """
    symbol_options = []

    for el in pattern_elements:
        if "Shift" in el:
            shift = el["Shift"]
            symbol = "-" if shift.upper() == "OFF" else shift
            symbol_options.append([symbol])
        elif "ShiftGroup" in el:
            group_id = el["ShiftGroup"]
            group_shifts = shift_groups.get(group_id, [])
            group_symbols = ["-" if s.upper() == "OFF" else s for s in group_shifts]
            symbol_options.append(group_symbols)
        else:
            # Se o XML contiver algo inesperado
            symbol_options.append(["?"])

    # Produto cartesiano para formar todas as combinações possíveis
    combinations = itertools.product(*symbol_options)
    expanded = ["".join(seq) for seq in combinations]
    return expanded


def load_data(xml_path: str ='ORTEC01.xml'):
    import xml.etree.ElementTree as ET
    from collections import defaultdict
    from datetime import datetime, timedelta

    # Função auxiliar para converter datas no formato ISO yyyy-mm-dd
    def parse_date(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    
    def calculate_duration(start_time, end_time):
        if not start_time or not end_time:
            return 0.0
        start = datetime.strptime(start_time, "%H:%M:%S")
        end = datetime.strptime(end_time, "%H:%M:%S")
        if end < start:
            end += timedelta(days=1)  # turno cruza meia-noite
        return (end - start).total_seconds() / 3600

    # Parse do XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Dicionários para armazenar dados
    shifts = {}                     # ID do turno -> informações
    employees = {}                  # ID do funcionário -> nome e contrato
    contracts = defaultdict(list)   # ID do contrato -> lista de padrões/regras
    cover_requirements = defaultdict(dict)  # Dia da semana -> turnos -> demanda
    shift_off_requests = []         # Lista de pedidos de folga
    shift_on_requests = []          # Lista de pedidos de turno desejado
    shift_groups = {}
    cover_weights = {}

    start_date = parse_date(root.findtext("StartDate"))
    end_date = parse_date(root.findtext("EndDate"))
    
    # Turnos disponíveis
    for shift in root.find("ShiftTypes"):
        shift_id = shift.attrib["ID"]
        # Converter horários para strings (ou datetime.time se quiser)
        start_time = shift.findtext("StartTime")
        end_time = shift.findtext("EndTime")

        # Se quiser converter para time, descomente abaixo:
        # from datetime import time
        # start_time = datetime.strptime(start_time, "%H:%M:%S").time() if start_time else None
        # end_time = datetime.strptime(end_time, "%H:%M:%S").time() if end_time else None

        shifts[shift_id] = {
            "Label": shift.findtext("Label"),
            "Name": shift.findtext("Name"),
            "StartTime": start_time,
            "EndTime": end_time,
            "Color": shift.findtext("Color")
        }
        shifts[shift_id]["Duration"] = calculate_duration(start_time, end_time)

    # Adicionar turno especial para folga ("OFF")
    # Se o turno "OFF" não estiver nos dados, adicionamos.
    if "OFF" not in shifts:
        shifts["OFF"] = {
            "Label": "OFF",
            "Name": "-",
            "StartTime": None,
            "EndTime": None,
            "Duration": 0.0,
        }
    if root.find("ShiftGroups") is not None:
     # ShiftGroups
        for group in root.find("ShiftGroups"):
            group_id = group.attrib["ID"]
            shift_list = [s.text for s in group.findall("Shift")]
            shift_groups[group_id] = shift_list

    # Funcionários
    for emp in root.find("Employees"):
        emp_id = emp.attrib["ID"]
        employees[emp_id] = {
            "Name": emp.findtext("Name"),
            "ContractID": emp.findtext("ContractID")  # Pode converter para int se quiser
        }

    # Contratos e padrões
    for contract in root.find("Contracts"):
        contract_id = contract.attrib["ID"]
        rules = []

        patterns_node = contract.find("Patterns")
        if patterns_node is not None:
            for match in patterns_node.findall("Match"):
                max_elem = match.find("Max")
                max_weight_elem = max_elem.find("Weight") if max_elem is not None else None

                max_weight_value = float(max_weight_elem.text) if max_weight_elem is not None else None
                max_weight_function = max_weight_elem.attrib.get("function") if max_weight_elem is not None else None
                max_count = int(max_elem.findtext("Count")) if max_elem is not None else None
                max_label = max_elem.findtext("Label") if max_elem is not None else None

                min_elem = match.find("Min")
                if min_elem is not None:
                    min_weight_elem = min_elem.find("Weight")
                    min_weight_value = float(min_weight_elem.text) if min_weight_elem is not None else None
                    min_weight_function = min_weight_elem.attrib.get("function") if min_weight_elem is not None else None
                    min_count = int(min_elem.findtext("Count"))
                    min_label = min_elem.findtext("Label")
                    min_dict = {
                        "Count": min_count,
                        "Weight": min_weight_value,
                        "WeightFunction": min_weight_function,
                        "Label": min_label
                    }
                else:
                    min_dict = None

                region_start = int(match.findtext("RegionStart")) if match.findtext("RegionStart") else None
                region_end = int(match.findtext("RegionEnd")) if match.findtext("RegionEnd") else None

                rule = {
                    "Max": {
                        "Count": max_count,
                        "Weight": max_weight_value,
                        "WeightFunction": max_weight_function,
                        "Label": max_label
                    },
                    "Min": min_dict,
                    "RegionStart": region_start,
                    "RegionEnd": region_end,
                    "Pattern": []
                }

                for pat in match.findall("Pattern"):
                    pat_dict = {}
                    for el in pat:
                        pat_dict[el.tag] = el.text
                    rule["Pattern"].append(pat_dict)

                rule["ExpandedPatterns"] = expand_pattern(rule["Pattern"], shift_groups)

                rules.append(rule)

        contracts[contract_id] = rules


    # Requisitos de cobertura por dia da semana
    for day_cover in root.find("CoverRequirements").findall("DayOfWeekCover"):
        day = day_cover.findtext("Day")
        for cover in day_cover.findall("Cover"):
            shift_id = cover.findtext("Shift")
            preferred = int(cover.findtext("Preferred"))
            cover_requirements[day][shift_id] = preferred

    # Pedidos de folga (ShiftOff)
    for req in root.find("ShiftOffRequests").findall("ShiftOff"):
        emp_id = req.findtext("EmployeeID")
        shift_type = req.findtext("ShiftTypeID") or req.findtext("Shift")  # usa ShiftTypeID se tiver, senão Shift
        weight = int(req.attrib.get("weight", 1))

        # Tenta obter a data completa primeiro
        date_str = req.findtext("Date")
        if date_str:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            # Se não tiver Date, usa Day (inteiro) a partir da start_date
            day_number = req.findtext("Day")
            if day_number is None:
                continue  # ignora se não houver nem Date nem Day
            day_index = int(day_number)
            date = start_date + timedelta(days=day_index)

        shift_off_requests.append({
            "EmployeeID": emp_id,
            "ShiftTypeID": shift_type,
            "Date": date,
            "Weight": weight
        })
        
    # Pedidos de turno desejado (ShiftOn)
    for req in root.find("ShiftOnRequests").findall("ShiftOn"):
        emp_id = req.findtext("EmployeeID")
        shift_type = req.findtext("ShiftTypeID") or req.findtext("Shift")
        weight = int(req.attrib.get("weight", 1))

        # Tenta obter a data completa primeiro
        date_str = req.findtext("Date")
        if date_str:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            day_number = req.findtext("Day")
            if day_number is None:
                continue  # ignora se não houver nem Date nem Day
            day_index = int(day_number)
            date = start_date + timedelta(days=day_index)

        shift_on_requests.append({
            "EmployeeID": emp_id,
            "ShiftTypeID": shift_type,
            "Date": date,
            "Weight": weight
        })

    # CoverWeights
    cover_weights_node = root.find("CoverWeights")
    if cover_weights_node is not None:
        cover_weights["PrefOverStaffing"] = int(cover_weights_node.findtext("PrefOverStaffing", default="0"))
        cover_weights["PrefUnderStaffing"] = int(cover_weights_node.findtext("PrefUnderStaffing", default="0"))


    

    return {
        "shifts": shifts,
        "shift_groups" : shift_groups,
        "employees": employees,
        "contracts": contracts,
        "cover_requirements": cover_requirements,
        "shift_off_requests": shift_off_requests,
        "shift_on_requests": shift_on_requests,
        "start_date": start_date,
        "end_date": end_date,
        "cover_weights" : cover_weights,
    }
